/**
 * @file event_system.hpp
 * @brief Event-driven execution system for TT-Metal
 *
 * Implements millisecond-scale temporal control using hardware
 * synchronization primitives and software timers.
 *
 * @author GRYPHGEN Project
 * @date 2025
 * @license Apache 2.0
 */

#pragma once

#include <functional>
#include <chrono>
#include <queue>
#include <memory>
#include <atomic>

namespace dynamic_cortex {
namespace tt_metal_backend {

using namespace std::chrono;
using EventAction = std::function<void()>;
using Timestamp = high_resolution_clock::time_point;

/**
 * @brief Event types for different trigger mechanisms
 */
enum class EventType {
    TIMER,           ///< Periodic or one-shot timer
    HARDWARE,        ///< Hardware synchronization signal
    SOFTWARE,        ///< Software-triggered event
    CONDITIONAL      ///< Condition-based event
};

/**
 * @brief Event priority levels
 */
enum class EventPriority {
    CRITICAL = 0,    ///< Highest priority (e.g., safety limits)
    HIGH = 1,        ///< High priority (e.g., timing-critical operations)
    NORMAL = 2,      ///< Normal priority (default)
    LOW = 3          ///< Low priority (background tasks)
};

/**
 * @brief Base event class
 */
class EventBase {
public:
    EventBase(const std::string& name, EventType type, EventPriority priority);
    virtual ~EventBase() = default;

    // Identity
    const std::string& name() const { return name_; }
    EventType type() const { return type_; }
    EventPriority priority() const { return priority_; }
    uint64_t id() const { return id_; }

    // Execution
    virtual void trigger() = 0;
    virtual bool shouldTrigger(Timestamp current_time) const = 0;

    // Control
    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool isEnabled() const { return enabled_; }

    // Statistics
    uint64_t triggerCount() const { return trigger_count_; }
    Timestamp lastTriggerTime() const { return last_trigger_time_; }
    microseconds averageTriggerDuration() const;

protected:
    std::string name_;
    EventType type_;
    EventPriority priority_;
    uint64_t id_;
    bool enabled_;
    uint64_t trigger_count_;
    Timestamp last_trigger_time_;
    std::vector<microseconds> trigger_durations_;

private:
    static std::atomic<uint64_t> next_id_;
};

/**
 * @brief Timer-based event
 */
class TimerEvent : public EventBase {
public:
    TimerEvent(const std::string& name, milliseconds period,
               EventAction action, bool one_shot = false);

    void trigger() override;
    bool shouldTrigger(Timestamp current_time) const override;

    // Configuration
    void setPeriod(milliseconds period);
    void setOneShot(bool one_shot);
    void setAction(EventAction action);

    // Timer control
    void reset();
    void restart();

    milliseconds period() const { return period_; }
    bool isOneShot() const { return one_shot_; }

private:
    milliseconds period_;
    EventAction action_;
    bool one_shot_;
    Timestamp next_trigger_time_;
    bool has_triggered_;
};

/**
 * @brief Conditional event (triggers when predicate is true)
 */
class ConditionalEvent : public EventBase {
public:
    using Predicate = std::function<bool()>;

    ConditionalEvent(const std::string& name, Predicate predicate,
                     EventAction action);

    void trigger() override;
    bool shouldTrigger(Timestamp current_time) const override;

    void setPredicate(Predicate predicate);
    void setAction(EventAction action);

private:
    Predicate predicate_;
    EventAction action_;
    mutable Timestamp last_check_time_;
    static constexpr milliseconds check_interval_{1};  // Check every 1ms
};

/**
 * @brief Hardware synchronization event
 */
class HardwareSyncEvent : public EventBase {
public:
    HardwareSyncEvent(const std::string& name, EventAction action);

    void trigger() override;
    bool shouldTrigger(Timestamp current_time) const override;

    // Hardware signal interface
    void signalFromHardware();
    void waitForHardware(milliseconds timeout);
    bool isHardwareSignaled() const;

private:
    EventAction action_;
    std::atomic<bool> hw_signaled_;
    Timestamp hw_signal_time_;
};

/**
 * @brief Event scheduler with priority queue
 */
class EventScheduler {
public:
    EventScheduler();
    ~EventScheduler();

    // Event management
    void addEvent(std::shared_ptr<EventBase> event);
    void removeEvent(uint64_t event_id);
    void removeEvent(const std::string& event_name);
    void clearEvents();

    // Execution
    void start();
    void stop();
    void step(milliseconds timestep);
    void runFor(milliseconds duration);
    void runUntil(Timestamp end_time);

    // Status
    bool isRunning() const { return running_; }
    Timestamp currentTime() const { return current_time_; }
    Timestamp startTime() const { return start_time_; }
    size_t numActiveEvents() const;

    // Performance
    void setMaxEventsPerStep(size_t max_events);
    void enableRealtime(bool enable);  // Try to match wall-clock time
    microseconds averageStepDuration() const;

private:
    struct EventComparator {
        bool operator()(const std::shared_ptr<EventBase>& a,
                        const std::shared_ptr<EventBase>& b) const {
            return static_cast<int>(a->priority()) > static_cast<int>(b->priority());
        }
    };

    std::priority_queue<std::shared_ptr<EventBase>,
                        std::vector<std::shared_ptr<EventBase>>,
                        EventComparator> event_queue_;

    std::vector<std::shared_ptr<EventBase>> all_events_;
    std::atomic<bool> running_;
    Timestamp start_time_;
    Timestamp current_time_;
    size_t max_events_per_step_;
    bool realtime_mode_;
    std::vector<microseconds> step_durations_;

    void processEvents();
    void updateCurrentTime(milliseconds delta);
};

/**
 * @brief Behavioral context modulation event
 */
class ContextModulationEvent : public EventBase {
public:
    struct BehavioralContext {
        bool is_rewarded;
        bool is_go_trial;
        float attention_level;
        uint32_t trial_number;
    };

    using ContextCallback = std::function<void(const BehavioralContext&)>;

    ContextModulationEvent(const std::string& name);

    void trigger() override;
    bool shouldTrigger(Timestamp current_time) const override;

    // Context management
    void setContext(const BehavioralContext& context);
    const BehavioralContext& currentContext() const;

    // Callbacks
    void addCallback(ContextCallback callback);
    void clearCallbacks();

private:
    BehavioralContext current_context_;
    BehavioralContext pending_context_;
    std::vector<ContextCallback> callbacks_;
    std::atomic<bool> context_updated_;
};

/**
 * @brief Channel rotation event
 */
class ChannelRotationEvent : public TimerEvent {
public:
    using RotationCallback = std::function<void(uint32_t old_channel, uint32_t new_channel)>;

    ChannelRotationEvent(const std::string& name, milliseconds period,
                         uint32_t num_channels);

    void trigger() override;

    // Rotation control
    void setNumChannels(uint32_t num_channels);
    void setCurrentChannel(uint32_t channel);
    uint32_t getCurrentChannel() const;
    uint32_t getNextChannel() const;

    // Behavioral modulation
    void setRewardedMode(bool rewarded);
    bool isRewardedMode() const;

    // Callback
    void setRotationCallback(RotationCallback callback);

private:
    uint32_t num_channels_;
    uint32_t current_channel_;
    bool rewarded_mode_;
    RotationCallback callback_;

    void rotateChannel();
};

/**
 * @brief Stimulus presentation event
 */
class StimulusPresentationEvent : public EventBase {
public:
    struct StimulusInfo {
        float orientation;      // degrees
        float spatial_freq;     // cycles/degree
        float contrast;         // [0.0, 1.0]
        milliseconds duration;
    };

    using StimulusCallback = std::function<void(const StimulusInfo&)>;

    StimulusPresentationEvent(const std::string& name);

    void trigger() override;
    bool shouldTrigger(Timestamp current_time) const override;

    // Stimulus control
    void presentStimulus(const StimulusInfo& stimulus);
    void setCallback(StimulusCallback callback);

    const StimulusInfo& currentStimulus() const;
    bool isStimulusPresent() const;

private:
    StimulusInfo current_stimulus_;
    StimulusCallback callback_;
    Timestamp stimulus_start_time_;
    std::atomic<bool> stimulus_active_;
};

/**
 * @brief Event statistics collector
 */
class EventStatistics {
public:
    struct EventStats {
        uint64_t trigger_count;
        microseconds total_duration;
        microseconds avg_duration;
        microseconds min_duration;
        microseconds max_duration;
        Timestamp last_trigger;
    };

    void recordTrigger(uint64_t event_id, microseconds duration);
    EventStats getStats(uint64_t event_id) const;
    void reset();

    // Global statistics
    uint64_t totalTriggers() const;
    microseconds totalEventTime() const;
    float eventOverheadPercent(microseconds total_runtime) const;

private:
    std::unordered_map<uint64_t, EventStats> stats_map_;
    std::mutex stats_mutex_;
};

/**
 * @brief Utility functions for event system
 */
namespace event_utils {

/**
 * @brief Create a periodic channel rotation event
 * @param period Rotation period (e.g., 15ms for rewarded, 121ms for non-rewarded)
 * @param num_channels Number of communication channels
 * @param callback Callback function for rotation
 * @return Shared pointer to event
 */
std::shared_ptr<ChannelRotationEvent> createChannelRotation(
    milliseconds period,
    uint32_t num_channels,
    ChannelRotationEvent::RotationCallback callback);

/**
 * @brief Create a trial sequence event
 * @param trial_duration Duration of each trial
 * @param num_trials Number of trials (0 = infinite)
 * @param callback Callback at start of each trial
 * @return Shared pointer to event
 */
std::shared_ptr<TimerEvent> createTrialSequence(
    milliseconds trial_duration,
    uint32_t num_trials,
    EventAction callback);

/**
 * @brief Create behavioral context event
 * @param callback Context update callback
 * @return Shared pointer to event
 */
std::shared_ptr<ContextModulationEvent> createContextModulation(
    ContextModulationEvent::ContextCallback callback);

} // namespace event_utils

} // namespace tt_metal_backend
} // namespace dynamic_cortex
