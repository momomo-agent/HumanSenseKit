#if os(iOS)
import Foundation
import Observation

/// Layer 3: Semantic/Fluent API for condition-based actions
@MainActor
public class HumanSenseObserver {
    private let state: HumanSenseState

    /// Creates a new observer bound to the given state.
    /// - Parameter state: The human-sense state to observe.
    public init(state: HumanSenseState) {
        self.state = state
    }

    // MARK: - Fluent API

    /// Begins building a condition-based observation with a custom predicate.
    /// - Parameter condition: A closure evaluated against the current state.
    /// - Returns: A ``ConditionBuilder`` for chaining additional conditions.
    public func when(_ condition: @escaping @MainActor (HumanSenseState) -> Bool) -> ConditionBuilder {
        return ConditionBuilder(state: state, conditions: [condition])
    }

    // MARK: - Convenience conditions

    /// Observes when the user is looking at the screen.
    public func whenLookingAtScreen() -> ConditionBuilder {
        when { $0.isLookingAtScreen }
    }

    /// Observes when the user is speaking.
    public func whenSpeaking() -> ConditionBuilder {
        when { $0.isSpeaking }
    }

    /// Observes when the user is speaking while looking at the screen.
    public func whenSpeakingToDevice() -> ConditionBuilder {
        when { $0.isSpeakingToDevice }
    }

    /// Observes when the user is nodding.
    public func whenNodding() -> ConditionBuilder {
        when { $0.isNodding }
    }

    /// Observes when the user is shaking their head.
    public func whenShakingHead() -> ConditionBuilder {
        when { $0.isShakingHead }
    }

    /// Observes when a specific hand gesture is detected.
    /// - Parameter gesture: The hand gesture to watch for.
    public func whenHandGesture(_ gesture: HandGesture) -> ConditionBuilder {
        when { $0.currentHandGesture == gesture }
    }
}

// MARK: - Condition Builder

/// Chainable builder for composing observation conditions.
@MainActor
public class ConditionBuilder {
    private let state: HumanSenseState
    private let conditions: [@MainActor (HumanSenseState) -> Bool]

    init(state: HumanSenseState, conditions: [@MainActor (HumanSenseState) -> Bool]) {
        self.state = state
        self.conditions = conditions
    }

    /// Adds a custom condition that must also be true.
    /// - Parameter condition: A closure evaluated against the current state.
    /// - Returns: A new ``ConditionBuilder`` with the added condition.
    public func and(_ condition: @escaping @MainActor (HumanSenseState) -> Bool) -> ConditionBuilder {
        return ConditionBuilder(state: state, conditions: conditions + [condition])
    }

    /// Adds a "looking at screen" condition.
    public func andLookingAtScreen() -> ConditionBuilder {
        and { $0.isLookingAtScreen }
    }

    /// Adds a "speaking" condition.
    public func andSpeaking() -> ConditionBuilder {
        and { $0.isSpeaking }
    }

    /// Adds a "nodding" condition.
    public func andNodding() -> ConditionBuilder {
        and { $0.isNodding }
    }

    /// Start observing: calls action whenever all conditions are met
    @discardableResult
    public func then(_ action: @escaping @MainActor () -> Void) -> ObservationTask {
        let state = self.state
        let conditions = self.conditions
        let task = Task { @MainActor in
            while !Task.isCancelled {
                withObservationTracking {
                    let allTrue = conditions.allSatisfy { $0(state) }
                    if allTrue { action() }
                } onChange: {
                    // Will re-enter the loop
                }
                try? await Task.sleep(for: .milliseconds(16)) // ~60fps
            }
        }
        return ObservationTask(task: task)
    }
}

/// Cancellable observation handle
public final class ObservationTask: @unchecked Sendable {
    private let task: Task<Void, Never>

    init(task: Task<Void, Never>) {
        self.task = task
    }

    /// Cancels the observation, stopping future condition checks.
    public func cancel() {
        task.cancel()
    }

    deinit {
        task.cancel()
    }
}
#endif
