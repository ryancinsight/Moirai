use super::{
    WebPermissionKind, WebPermissionOutcome, error,
    registry::Registry,
    request::{Completion, malformed},
};
use std::{io, rc::Rc};
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{MediaDevices, MediaStream, MediaStreamConstraints, MediaStreamTrack, Window};

/// Acquired browser stream; drop stops every native track.
/// Track counts describe ownership, including tracks the browser already ended.
/// [Media Capture](https://www.w3.org/TR/mediacapture-streams/)'s track `stop`
/// algorithm ends each owned track; an unresolved acquisition has no abort API.
pub struct WebMediaLease {
    stream: MediaStream,
    kind: WebPermissionKind,
    tracks: usize,
    registry: Rc<Registry>,
}

impl WebMediaLease {
    fn acquired(stream: MediaStream, completion: &Completion) -> io::Result<Self> {
        let tracks = usize::try_from(stream.get_tracks().length())
            .map_err(|_| malformed("Browser media track count is not representable"));
        let registered = tracks.and_then(|tracks| {
            if tracks == 0 {
                return Err(malformed("Browser media grant has no tracks"));
            }
            completion
                .registry
                .state
                .borrow_mut()
                .media_acquired(tracks)?;
            Ok(tracks)
        });
        match registered {
            Ok(tracks) => Ok(Self {
                stream,
                kind: completion.token.kind,
                tracks,
                registry: Rc::clone(&completion.registry),
            }),
            Err(error) => {
                stop(&stream);
                Err(error)
            }
        }
    }

    /// Camera or microphone kind that created this lease.
    #[must_use]
    pub const fn kind(&self) -> WebPermissionKind {
        self.kind
    }

    /// Number of native tracks owned by this lease.
    #[must_use]
    pub const fn track_count(&self) -> usize {
        self.tracks
    }
}

fn stop(stream: &MediaStream) {
    for value in stream.get_tracks().iter() {
        // MediaStream.getTracks has the WebIDL result sequence<MediaStreamTrack>.
        // The stream itself comes from a checked actual getUserMedia result.
        value.unchecked_into::<MediaStreamTrack>().stop();
    }
}

impl Drop for WebMediaLease {
    fn drop(&mut self) {
        stop(&self.stream);
        self.registry.state.borrow_mut().media_released(self.tracks);
    }
}

pub(super) fn start(window: &Window, completion: Rc<Completion>) {
    let navigator = window.navigator();
    let devices = match error::interface(navigator.as_ref(), "mediaDevices") {
        Ok(Some(value)) => match value.dyn_into::<MediaDevices>() {
            Ok(devices) => devices,
            Err(_) => {
                completion.finish(Err(malformed(
                    "Browser mediaDevices has an invalid interface",
                )));
                return;
            }
        },
        Ok(None) => {
            completion.unavailable();
            return;
        }
        Err(error) => {
            completion.reject(error);
            return;
        }
    };
    match error::method(devices.as_ref(), "getUserMedia") {
        Ok(true) => (),
        Ok(false) => {
            completion.unavailable();
            return;
        }
        Err(error) => {
            completion.reject(error);
            return;
        }
    }
    let constraints = MediaStreamConstraints::new();
    // One acquisition algorithm; the admitted kind selects one constraint.
    constraints.set_video_bool(completion.token.kind == WebPermissionKind::Camera);
    constraints.set_audio_bool(completion.token.kind == WebPermissionKind::Microphone);
    let promise = match devices.get_user_media_with_constraints(&constraints) {
        Ok(promise) => promise,
        Err(value) => {
            completion.reject(error::native(&value));
            return;
        }
    };
    // This task owns genuine settlement, including after consumer cancellation.
    // It captures only Completion/Registry and Promise, never page DOM handles.
    wasm_bindgen_futures::spawn_local(async move {
        match JsFuture::from(promise).await {
            Ok(value) => {
                let result = value
                    .dyn_into::<MediaStream>()
                    .map_err(|_| malformed("Browser media grant is not a MediaStream"))
                    .and_then(|stream| WebMediaLease::acquired(stream, &completion))
                    .map(WebPermissionOutcome::MediaGranted);
                completion.finish(result);
            }
            Err(value) => completion.reject(error::native(&value)),
        }
    });
}
