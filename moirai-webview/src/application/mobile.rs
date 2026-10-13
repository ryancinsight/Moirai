//! Scene-owned UIKit lifecycle, including native geometry and reconnection.

use crate::{ResourceRoot, webview};
use objc2::{
    ClassType, DefinedClass, MainThreadOnly, define_class, msg_send,
    rc::{Allocated, Retained},
};
use objc2_foundation::{NSObject, NSObjectProtocol, NSString};
use objc2_ui_kit::{
    UIApplication, UIApplicationDelegate, UIScene, UISceneConfiguration, UISceneConnectionOptions,
    UISceneDelegate, UISceneSession, UIViewController, UIWindow, UIWindowScene,
    UIWindowSceneDelegate,
};
use std::{cell::RefCell, io};

thread_local! { static RESOURCES: RefCell<Option<(ResourceRoot, String)>> = const { RefCell::new(None) }; }

struct Mounted {
    _window: Retained<UIWindow>,
    _content: Retained<webview::Content>,
}

#[derive(Default)]
struct State {
    mounted: RefCell<Option<Mounted>>,
}

define_class!(
    // SAFETY: NSObject has no subclassing contract; UIKit constructs and
    // invokes this delegate on the main thread. Rust ivars initialize in init.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[ivars = State]
    struct SceneDelegate;

    impl SceneDelegate {
        #[unsafe(method_id(init))]
        fn init(this: Allocated<Self>) -> Retained<Self> {
            let this = this.set_ivars(State::default());
            // SAFETY: Complete ivars and NSObject's initializer.
            unsafe { msg_send![super(this), init] }
        }
        #[unsafe(method(scene:willConnectToSession:options:))]
        fn connected(&self, scene: &UIScene, _session: &UISceneSession, _options: &UISceneConnectionOptions) {
            if let Err(error) = self.setup(scene) {
                tracing::error!(%error, "Apple scene setup failed");
                std::process::exit(1);
            }
        }
        #[unsafe(method(sceneDidDisconnect:))]
        fn disconnected(&self, _scene: &UIScene) { self.ivars().mounted.borrow_mut().take(); }
    }
    // SAFETY: No additional NSObjectProtocol invariants.
    unsafe impl NSObjectProtocol for SceneDelegate {}
    // SAFETY: UIKit calls the declared scene selectors on the main thread.
    unsafe impl UISceneDelegate for SceneDelegate {}
    // SAFETY: The scene window retains its view controller and WebView.
    unsafe impl UIWindowSceneDelegate for SceneDelegate {}
);

define_class!(
    // SAFETY: This NSObject delegate carries no Rust ivars or subclass contract.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    struct Delegate;
    // SAFETY: No additional NSObjectProtocol invariants.
    unsafe impl NSObjectProtocol for Delegate {}
    // SAFETY: The callback uses UIApplicationDelegate's exact selector/signature.
    unsafe impl UIApplicationDelegate for Delegate {
        #[unsafe(method_id(application:configurationForConnectingSceneSession:options:))]
        fn configuration(
            &self,
            _app: &UIApplication,
            session: &UISceneSession,
            _options: &UISceneConnectionOptions,
        ) -> Retained<UISceneConfiguration> {
            let config = UISceneConfiguration::initWithName_sessionRole(
                UISceneConfiguration::alloc(self.mtm()),
                Some(&NSString::from_str("Metis")),
                &session.role(),
            );
            // SAFETY: SceneDelegate implements UIWindowSceneDelegate and its
            // init selector initializes every Rust ivar before callbacks.
            unsafe {
                config.setDelegateClass(Some(SceneDelegate::class()));
            }
            config
        }
    }
);

impl SceneDelegate {
    fn setup(&self, scene: &UIScene) -> io::Result<()> {
        let (root, title) = RESOURCES
            .with(|state| state.borrow().clone())
            .ok_or_else(|| io::Error::other("application resources are absent"))?;
        let scene = scene
            .downcast_ref::<UIWindowScene>()
            .ok_or_else(|| io::Error::other("application scene is not a window scene"))?;
        let mtm = self.mtm();
        let window = UIWindow::initWithWindowScene(UIWindow::alloc(mtm), scene);
        let content = webview::Content::new(root, mtm)?;
        let view = webview::create(&content, window.bounds())?;
        let controller = UIViewController::new(mtm);
        controller.setTitle(Some(&NSString::from_str(&title)));
        controller.setView(Some(&view));
        window.setRootViewController(Some(&controller));
        window.makeKeyAndVisible();
        self.ivars().mounted.replace(Some(Mounted {
            _window: window,
            _content: content,
        }));
        Ok(())
    }
}

pub(super) fn run(
    resources: ResourceRoot,
    title: &str,
    mtm: objc2::MainThreadMarker,
) -> io::Result<()> {
    RESOURCES.with(|state| {
        let mut state = state.borrow_mut();
        if state.is_some() {
            return Err(io::Error::other("Apple event loop is already configured"));
        }
        *state = Some((resources, title.to_owned()));
        Ok(())
    })?;
    UIApplication::main(None, Some(&NSString::from_class(Delegate::class())), mtm)
}
