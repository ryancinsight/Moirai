//! Shared WebKit construction, resource callbacks and navigation policy.

mod bindings;
pub(crate) use bindings::WebView;

use crate::ResourceRoot;
use objc2::{
    AnyThread, DefinedClass, MainThreadOnly, define_class, msg_send, rc::Retained,
    runtime::ProtocolObject,
};
use objc2_foundation::{
    NSData, NSError, NSObject, NSObjectProtocol, NSRect, NSString, NSURL, NSURLRequest,
    NSURLResponse,
};
use objc2_web_kit::{
    WKNavigation, WKNavigationAction, WKNavigationActionPolicy, WKNavigationDelegate,
    WKURLSchemeHandler, WKURLSchemeTask, WKWebViewConfiguration,
};
use std::io;

pub(crate) struct State {
    root: ResourceRoot,
}

define_class!(
    // SAFETY: NSObject has no subclassing requirements; every callback is
    // main-thread-only and the ivars outlive the native callbacks.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[ivars = State]
    pub(crate) struct Content;

    // SAFETY: NSObjectProtocol imposes no additional invariants.
    unsafe impl NSObjectProtocol for Content {}
    // SAFETY: These selectors implement the system protocol on macOS and iOS;
    // resource requests finish synchronously before control returns to WebKit.
    unsafe impl WKURLSchemeHandler for Content {}
    // SAFETY: The policy callback is invoked exactly once per navigation.
    unsafe impl WKNavigationDelegate for Content {}

    impl Content {
        #[unsafe(method(webView:didFinishNavigation:))]
        fn finished(&self, _view: &WebView, _navigation: Option<&WKNavigation>) {
            tracing::info!("packaged document navigation finished");
        }

        #[unsafe(method(webView:didFailProvisionalNavigation:withError:))]
        fn failed(&self, _view: &WebView, _navigation: Option<&WKNavigation>, error: &NSError) {
            tracing::error!(description = %error.localizedDescription(), "packaged document navigation failed");
        }

        #[unsafe(method(webView:startURLSchemeTask:))]
        fn start(&self, _view: &WebView, task: &ProtocolObject<dyn WKURLSchemeTask>) {
            if let Err(error) = self.deliver(task) {
                let message = NSString::from_str(&format!("packaged resource: {error}"));
                tracing::error!(%error, "packaged resource delivery failed");
                // SAFETY: The domain is a live string and no user-info object
                // is borrowed across this synchronous error callback.
                let native = unsafe { NSError::errorWithDomain_code_userInfo(&message, 1, None) };
                // SAFETY: This synchronous task has received no completion.
                unsafe { task.didFailWithError(&native) };
            }
        }

        #[unsafe(method(webView:stopURLSchemeTask:))]
        fn stop(&self, _view: &WebView, _task: &ProtocolObject<dyn WKURLSchemeTask>) {
            // Delivery is synchronous on the same thread. No task is retained,
            // so a stop cannot leave an outstanding callback or queued work.
        }

        #[unsafe(method(webView:decidePolicyForNavigationAction:decisionHandler:))]
        fn decide(&self, _view: &WebView, action: &WKNavigationAction, handler: &block2::DynBlock<dyn Fn(WKNavigationActionPolicy)>) {
            // SAFETY: WebKit owns the live navigation action for this callback.
            let allowed = unsafe { action.request() }.URL().and_then(|url| url.absoluteString()).is_some_and(|uri| ResourceRoot::permits(&uri.to_string()));
            handler.call((if allowed { WKNavigationActionPolicy::Allow } else { WKNavigationActionPolicy::Cancel },));
        }
    }
);

impl Content {
    pub(crate) fn new(root: ResourceRoot, mtm: objc2::MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(State { root });
        // SAFETY: Initialized NSObject subclass with complete Rust ivars.
        unsafe { msg_send![super(this), init] }
    }

    fn deliver(&self, task: &ProtocolObject<dyn WKURLSchemeTask>) -> io::Result<()> {
        // SAFETY: The native callback lends a live, not-yet-completed task.
        let request = unsafe { task.request() };
        let url = request.URL().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "resource request has no URL")
        })?;
        let uri = url.absoluteString().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "resource URL has no string")
        })?;
        let (bytes, mime) = self.ivars().root.read(&uri.to_string())?;
        tracing::info!(
            content_type = mime,
            bytes = bytes.len(),
            "packaged resource delivered"
        );
        let length = isize::try_from(bytes.len()).map_err(io::Error::other)?;
        let response = NSURLResponse::initWithURL_MIMEType_expectedContentLength_textEncodingName(
            NSURLResponse::alloc(),
            &url,
            Some(&NSString::from_str(mime)),
            length,
            None,
        );
        let data = NSData::with_bytes(&bytes);
        // SAFETY: The synchronous task receives response, data, completion in
        // order and cannot be cancelled while this callback owns the thread.
        unsafe {
            task.didReceiveResponse(&response);
            task.didReceiveData(&data);
            task.didFinish();
        }
        Ok(())
    }
}

pub(crate) fn create(content: &Content, frame: NSRect) -> io::Result<Retained<WebView>> {
    // SAFETY: The main-thread marker comes from the live content delegate.
    let config = unsafe { WKWebViewConfiguration::new(content.mtm()) };
    // SAFETY: WebKit retains the live scheme handler and the configured scheme
    // is custom, rather than overriding a built-in network scheme.
    unsafe {
        config.setURLSchemeHandler_forURLScheme(
            Some(ProtocolObject::from_ref(content)),
            &NSString::from_str("metis"),
        );
    }
    let view = WebView::init(WebView::alloc(content.mtm()), frame, &config);
    view.set_delegate(Some(ProtocolObject::from_ref(content)));
    let url = NSURL::URLWithString(&NSString::from_str("metis://app/index.html"))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid packaged entry URL"))?;
    let request = NSURLRequest::requestWithURL(&url);
    view.load(&request)
        .ok_or_else(|| io::Error::other("WebKit refused entry navigation"))?;
    Ok(view)
}
