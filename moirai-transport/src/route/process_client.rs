use super::{
    ProcessEndpoint, RoutedProcessTaskError, RoutedProcessTaskOutput, RoutedProcessTaskResult,
};
use crate::process::{ProcessDropPolicy, ProcessError, ProcessSupervisor, ProcessWaitPolicy};
use crate::remote_task::{RemoteTaskClient, RemoteTaskId, RemoteTaskOperation};
use crate::RemoteAddress;
use moirai_core::Priority;
use moirai_executor::schedule::{RoutePolicy, SchedulerRoute, WorkClass};
use std::marker::PhantomData;

/// Remote task client that launches a supervised child process for process routes.
pub struct RoutedProcessTaskClient<P: RoutePolicy> {
    endpoints: Vec<ProcessEndpoint>,
    reply_to: RemoteAddress,
    drop_policy: ProcessDropPolicy,
    wait_policy: ProcessWaitPolicy,
    _policy: PhantomData<P>,
}

impl<P: RoutePolicy> RoutedProcessTaskClient<P> {
    /// Construct a routed process task client.
    pub fn new(
        endpoints: Vec<ProcessEndpoint>,
        reply_to: RemoteAddress,
        drop_policy: ProcessDropPolicy,
        wait_policy: ProcessWaitPolicy,
    ) -> Self {
        Self {
            endpoints,
            reply_to,
            drop_policy,
            wait_policy,
            _policy: PhantomData,
        }
    }

    /// Execute a fixed-format remote task through a selected process route.
    pub fn execute_route(
        &self,
        route: SchedulerRoute,
        task_id: RemoteTaskId,
        operation: RemoteTaskOperation,
    ) -> RoutedProcessTaskResult<RoutedProcessTaskOutput> {
        let SchedulerRoute::Process(route) = route else {
            return Err(RoutedProcessTaskError::NonProcessRoute);
        };

        let endpoint = self
            .endpoints
            .iter()
            .find(|endpoint| endpoint.process == route.process)
            .ok_or(RoutedProcessTaskError::MissingProcessEndpoint)?;
        let supervisor = ProcessSupervisor::new();
        let mut process = supervisor.spawn(endpoint.spec.clone(), self.drop_policy)?;
        let process_id = process.id();
        let result = RemoteTaskClient::new(endpoint.task_server.clone(), self.reply_to.clone())
            .execute(task_id, operation)?;
        let status = match process.wait_bounded(self.wait_policy)? {
            Some(status) => status,
            None => process
                .terminate()?
                .ok_or(RoutedProcessTaskError::Process(ProcessError::WaitFailed))?,
        };

        Ok(RoutedProcessTaskOutput {
            process_id,
            result,
            status,
        })
    }

    /// Select a process route and execute a fixed-format remote task through it.
    pub fn execute_selected<C>(
        &self,
        router: &moirai_executor::schedule::HybridRouter<P>,
        priority: Priority,
        sequence: usize,
        task_id: RemoteTaskId,
        operation: RemoteTaskOperation,
    ) -> RoutedProcessTaskResult<(SchedulerRoute, RoutedProcessTaskOutput)>
    where
        C: WorkClass,
    {
        let route = router.select::<C>(priority, sequence);
        let output = self.execute_route(route, task_id, operation)?;
        Ok((route, output))
    }
}
