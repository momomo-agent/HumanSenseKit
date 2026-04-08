import Foundation
import Combine

/// Layer 3: Semantic/Fluent API for condition-based actions
@MainActor
public class HumanSenseObserver {
    private let state: HumanSenseState
    private var cancellables = Set<AnyCancellable>()
    
    public init(state: HumanSenseState) {
        self.state = state
    }
    
    // MARK: - Fluent API
    
    public func when(_ condition: @escaping (HumanSenseState) -> Bool) -> ConditionBuilder {
        return ConditionBuilder(state: state, conditions: [condition])
    }
    
    // MARK: - Convenience conditions
    
    public func whenLookingAtScreen() -> ConditionBuilder {
        when { $0.isLookingAtScreen }
    }
    
    public func whenSpeaking() -> ConditionBuilder {
        when { $0.isSpeaking }
    }
    
    public func whenSpeakingToDevice() -> ConditionBuilder {
        when { $0.isSpeakingToDevice }
    }
    
    public func whenNodding() -> ConditionBuilder {
        when { $0.isNodding }
    }
    
    public func whenShakingHead() -> ConditionBuilder {
        when { $0.isShakingHead }
    }
    
    public func whenHandGesture(_ gesture: HandGesture) -> ConditionBuilder {
        when { $0.currentHandGesture == gesture }
    }
}

// MARK: - Condition Builder

public class ConditionBuilder {
    private let state: HumanSenseState
    private let conditions: [(HumanSenseState) -> Bool]
    
    init(state: HumanSenseState, conditions: [(HumanSenseState) -> Bool]) {
        self.state = state
        self.conditions = conditions
    }
    
    public func and(_ condition: @escaping (HumanSenseState) -> Bool) -> ConditionBuilder {
        return ConditionBuilder(state: state, conditions: conditions + [condition])
    }
    
    public func andLookingAtScreen() -> ConditionBuilder {
        and { $0.isLookingAtScreen }
    }
    
    public func andSpeaking() -> ConditionBuilder {
        and { $0.isSpeaking }
    }
    
    public func andNodding() -> ConditionBuilder {
        and { $0.isNodding }
    }
    
    public func then(_ action: @escaping () -> Void) -> AnyCancellable {
        state.$rawState
            .sink { [weak state] _ in
                guard let state = state else { return }
                let allTrue = self.conditions.allSatisfy { $0(state) }
                if allTrue { action() }
            }
    }
    
    public func onChange(_ action: @escaping (Bool) -> Void) -> AnyCancellable {
        var previousMatch = false
        return state.$rawState
            .sink { [weak state] _ in
                guard let state = state else { return }
                let currentMatch = self.conditions.allSatisfy { $0(state) }
                if currentMatch != previousMatch {
                    action(currentMatch)
                    previousMatch = currentMatch
                }
            }
    }
}
