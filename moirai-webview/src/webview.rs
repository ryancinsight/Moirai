//! Shared WebKit construction, resource callbacks and navigation policy.

mod bindings;
mod external;
mod navigation;
mod response;
pub(crate) use bindings::WebView;

use crate::{ExternalLinks, ResourceRoot, navigation::Decision};
use objc2::{
    DefinedClass, MainThreadOnly, define_class, msg_send, rc::Retained, runtime::ProtocolObject,
};
use objc2_foundation::{
    NSData, NSError, NSObject, NSObjectProtocol, NSRect, NSString, NSURL, NSURLRequest,
};
use objc2_web_kit::{
    WKNavigation, WKNavigationAction, WKNavigationActionPolicy, WKNavigationDelegate, WKUIDelegate,
    WKURLSchemeHandler, WKURLSchemeTask, WKWebViewConfiguration, WKWindowFeatures,
};
use std::io;

pub(crate) struct State {
    root: ResourceRoot,
    links: ExternalLinks,
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
    unsafe impl WKURLSchemeHandler for Content {
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
    }

    // SAFETY: Each callback has the exact system protocol ABI and the policy
    // handler is called exactly once per navigation.
    unsafe impl WKNavigationDelegate for Content {
        #[unsafe(method(webView:didFinishNavigation:))]
        fn finished(&self, _view: &WebView, _navigation: Option<&WKNavigation>) {
            tracing::info!("packaged document navigation finished");
        }

        #[unsafe(method(webView:didFailProvisionalNavigation:withError:))]
        fn failed(&self, _view: &WebView, _navigation: Option<&WKNavigation>, error: &NSError) {
            tracing::error!(description = %error.localizedDescription(), "packaged document navigation failed");
        }

        #[unsafe(method(webView:decidePolicyForNavigationAction:decisionHandler:))]
        fn decide(
            &self,
            _view: &WebView,
            action: &WKNavigationAction,
            handler: &block2::DynBlock<dyn Fn(WKNavigationActionPolicy)>,
        ) {
            let allowed = navigation::decision(&self.ivars().links, action) != Decision::Deny;
            // ExternalPopup only admits WebKit's UI callback; it does not
            // navigate an existing frame or open the destination at this stage.
            handler.call((if allowed {
                WKNavigationActionPolicy::Allow
            } else {
                WKNavigationActionPolicy::Cancel
            },));
        }
    }

    // SAFETY: The popup callback has WKUIDelegate's exact Objective-C ABI;
    // the retained Content outlives both weak native delegate references.
    unsafe impl WKUIDelegate for Content {
        #[unsafe(method_id(webView:createWebViewWithConfiguration:forNavigationAction:windowFeatures:))]
        fn popup(
            &self,
            _view: &WebView,
            _configuration: &WKWebViewConfiguration,
            action: &WKNavigationAction,
            _features: &WKWindowFeatures,
        ) -> Option<Retained<WebView>> {
            if navigation::decision(&self.ivars().links, action) == Decision::ExternalPopup {
                // SAFETY: WebKit lends the request during this synchronous callback.
                if let Some(url) = unsafe { action.request() }.URL()
                    && let Err(error) = external::open(&url, self.mtm())
                {
                    tracing::error!(%error, "native external link handoff failed");
                }
            }
            // The operating system owns the external destination. WebKit must
            // not create a remote frame inside this packaged application.
            None
        }
    }
);

impl Content {
    pub(crate) fn new(
        root: ResourceRoot,
        mtm: objc2::MainThreadMarker,
    ) -> io::Result<Retained<Self>> {
        let links = external::bundle_links()?;
        let this = Self::alloc(mtm).set_ivars(State { root, links });
        // SAFETY: Initialized NSObject subclass with complete Rust ivars.
        Ok(unsafe { msg_send![super(this), init] })
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
        let response = response::new(&url, mime, bytes.len())?;
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
    // SAFETY: These main-thread preferences are owned by the live configuration.
    // LinkActivated is a navigation classification, not a trusted-event proof.
    // WebKit's automatic-window restriction remains enabled.
    unsafe {
        config
            .preferences()
            .setJavaScriptCanOpenWindowsAutomatically(false)
    };
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
    view.set_ui_delegate(Some(ProtocolObject::from_ref(content)));
    let url = NSURL::URLWithString(&NSString::from_str("metis://app/index.html"))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid packaged entry URL"))?;
    let request = NSURLRequest::requestWithURL(&url);
    view.load(&request)
        .ok_or_else(|| io::Error::other("WebKit refused entry navigation"))?;
    Ok(view)
}
