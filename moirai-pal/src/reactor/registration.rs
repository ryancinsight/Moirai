use std::io;

#[cfg(any(unix, windows))]
use std::collections::HashMap;
#[cfg(any(unix, windows))]
use std::hash::Hash;

#[cfg(windows)]
use super::socket_owner::WeakSocketOwner;
#[cfg(any(unix, windows))]
use crate::Event;
use crate::Interest;

#[cfg(any(unix, windows))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct RegistrationGeneration(usize);

#[cfg(any(unix, windows))]
impl RegistrationGeneration {
    #[cfg(unix)]
    pub(crate) const fn get(self) -> usize {
        self.0
    }

    #[cfg(unix)]
    pub(crate) const fn from_raw(raw: usize) -> Option<Self> {
        if raw == 0 { None } else { Some(Self(raw)) }
    }
}

#[cfg(windows)]
#[derive(Clone, Copy)]
pub(crate) struct WaiterRegistration {
    pub(crate) generation: RegistrationGeneration,
    pub(crate) replaced_existing: bool,
}

#[cfg(any(unix, windows))]
#[derive(Clone)]
#[cfg_attr(unix, derive(Copy))]
pub(crate) struct Registration {
    pub(crate) interest: Interest,
    pub(crate) generation: RegistrationGeneration,
    #[cfg(windows)]
    pub(crate) owner: Option<WeakSocketOwner>,
}

#[cfg(any(unix, windows))]
pub(crate) struct RegistrationTable<K> {
    entries: HashMap<K, Registration>,
    #[cfg(target_os = "linux")]
    generations: HashMap<RegistrationGeneration, K>,
    next_generation: usize,
}

#[cfg(any(unix, windows))]
impl<K> Default for RegistrationTable<K> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            #[cfg(target_os = "linux")]
            generations: HashMap::new(),
            next_generation: 0,
        }
    }
}

#[cfg(any(unix, windows))]
impl<K> RegistrationTable<K>
where
    K: Copy + Eq + Hash,
{
    pub(crate) fn issue_generation(&mut self) -> io::Result<RegistrationGeneration> {
        let next_generation = self
            .next_generation
            .checked_add(1)
            .ok_or_else(|| io::Error::other("reactor registration generation exhausted"))?;
        self.next_generation = next_generation;
        Ok(RegistrationGeneration(next_generation))
    }

    pub(crate) fn commit(
        &mut self,
        key: K,
        interest: Interest,
        generation: RegistrationGeneration,
    ) {
        self.commit_registration(
            key,
            Registration {
                interest,
                generation,
                #[cfg(windows)]
                owner: None,
            },
        );
    }

    #[cfg(windows)]
    pub(crate) fn commit_owned(
        &mut self,
        key: K,
        interest: Interest,
        generation: RegistrationGeneration,
        owner: WeakSocketOwner,
    ) {
        self.commit_registration(
            key,
            Registration {
                interest,
                generation,
                owner: Some(owner),
            },
        );
    }

    fn commit_registration(&mut self, key: K, registration: Registration) {
        #[cfg(target_os = "linux")]
        let generation = registration.generation;
        #[cfg(target_os = "linux")]
        if let Some(previous) = self.entries.insert(key, registration) {
            self.generations.remove(&previous.generation);
        }
        #[cfg(not(target_os = "linux"))]
        let _ = self.entries.insert(key, registration);
        #[cfg(target_os = "linux")]
        self.generations.insert(generation, key);
    }

    pub(crate) fn get(&self, key: K) -> Option<Registration> {
        self.entries.get(&key).cloned()
    }

    #[cfg(windows)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(windows)]
    pub(crate) fn retain(&mut self, mut keep: impl FnMut(K, &Registration) -> bool) {
        self.entries
            .retain(|key, registration| keep(*key, registration));
    }

    #[cfg(target_os = "linux")]
    pub(crate) fn key_for_generation(&self, generation: RegistrationGeneration) -> Option<K> {
        self.generations.get(&generation).copied()
    }

    pub(crate) fn is_current(&self, key: K, generation: RegistrationGeneration) -> bool {
        self.entries
            .get(&key)
            .is_some_and(|registration| registration.generation == generation)
    }

    pub(crate) fn update_interest(
        &mut self,
        key: K,
        generation: RegistrationGeneration,
        interest: Interest,
    ) -> bool {
        let Some(registration) = self.entries.get_mut(&key) else {
            return false;
        };
        if registration.generation != generation {
            return false;
        }
        registration.interest = interest;
        true
    }

    pub(crate) fn remove(&mut self, key: K) -> Option<Registration> {
        let registration = self.entries.remove(&key)?;
        #[cfg(target_os = "linux")]
        self.generations.remove(&registration.generation);
        Some(registration)
    }

    #[cfg(windows)]
    pub(crate) fn remove_if_current(&mut self, key: K, generation: RegistrationGeneration) -> bool {
        if !self.is_current(key, generation) {
            return false;
        }
        self.remove(key).is_some()
    }

    #[cfg(all(test, windows))]
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(any(unix, windows))]
pub(crate) struct PolledEvent {
    event: Event,
    generation: RegistrationGeneration,
    #[cfg(windows)]
    invalidated: bool,
}

#[cfg(any(unix, windows))]
impl PolledEvent {
    pub(crate) const fn new(event: Event, generation: RegistrationGeneration) -> Self {
        Self {
            event,
            generation,
            #[cfg(windows)]
            invalidated: false,
        }
    }

    #[cfg(windows)]
    pub(crate) const fn invalidated(event: Event, generation: RegistrationGeneration) -> Self {
        Self {
            event,
            generation,
            invalidated: true,
        }
    }

    pub(crate) const fn event(&self) -> &Event {
        &self.event
    }

    pub(crate) const fn generation(&self) -> RegistrationGeneration {
        self.generation
    }

    #[cfg(windows)]
    pub(crate) const fn was_invalidated(&self) -> bool {
        self.invalidated
    }

    #[cfg(test)]
    pub(crate) const fn descriptor(&self) -> crate::RawFd {
        self.event.fd
    }
}

/// Failed platform transition paired with its observed postcondition.
///
/// `armed_interest` is the exact interest the backend still tracks after the
/// failure. `None` means the backend registration is absent.
pub(crate) struct PlatformUpdateFailure {
    error: io::Error,
    armed_interest: Option<Interest>,
}

impl PlatformUpdateFailure {
    pub(crate) const fn new(error: io::Error, armed_interest: Option<Interest>) -> Self {
        Self {
            error,
            armed_interest,
        }
    }

    pub(crate) const fn armed_interest(&self) -> Option<Interest> {
        self.armed_interest
    }

    pub(crate) fn into_error(self) -> io::Error {
        self.error
    }
}
