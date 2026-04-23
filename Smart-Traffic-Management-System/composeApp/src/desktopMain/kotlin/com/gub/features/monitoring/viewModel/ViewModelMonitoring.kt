package com.gub.features.monitoring.viewModel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.gub.core.domain.Response
import com.gub.core.logic.SignalingAlgorithm.getNextPhaseBasedOnData
import com.gub.domain.models.monitoring.ModelLiveSignal
import com.gub.features.monitoring.di.MonitoringModule
import com.gub.features.monitoring.domain.usecase.UseCaseLiveSignal
import com.gub.features.monitoring.presentation.components.TrafficPhase
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

class ViewModelMonitoring(
    private val liveSignal: UseCaseLiveSignal = MonitoringModule.getUseCaseLiveSignal
) : ViewModel() {

    private val _uiState = MutableStateFlow(MonitoringUiState())
    val uiState: StateFlow<MonitoringUiState> = _uiState.asStateFlow()

    private var signalTimerJob: Job? = null
    private var greenDurations = mutableMapOf(
        TrafficPhase.NS_GREEN to mutableListOf<Int>(),
        TrafficPhase.EW_GREEN to mutableListOf<Int>()
    )

    init {
        startSignalTimer()
        getLiveTrafficSignal()
    }

    private fun getLiveTrafficSignal() = viewModelScope.launch {
        when (val res = liveSignal()) {
            is Response.Error -> {}
            Response.Loading -> {}
            is Response.Success -> {
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    liveSignal = res.data,
                    error = null
                )
            }
        }
    }

    private fun startSignalTimer() {
        signalTimerJob?.cancel()
        signalTimerJob = viewModelScope.launch {
            while (isActive) {
                delay(1000)
                _uiState.update { state ->
                    val newTime = state.timeRemaining - 1
                    if (newTime > 0) {
                        state.copy(timeRemaining = newTime)
                    } else {
                        // Track green phase duration
                        if (state.currentPhase == TrafficPhase.NS_GREEN || state.currentPhase == TrafficPhase.EW_GREEN) {
                            greenDurations[state.currentPhase]?.add(state.originalGreenTime)
                        }

                        // Move to next phase
                        val nextPhaseWithTime = getNextPhaseBasedOnData(state)

                        state.copy(
                            currentPhase = nextPhaseWithTime.first,
                            timeRemaining = nextPhaseWithTime.second,
                            originalGreenTime = nextPhaseWithTime.second,
                            averageNSGreen = computeAverage(TrafficPhase.NS_GREEN),
                            averageEWGreen = computeAverage(TrafficPhase.EW_GREEN)
                        )
                    }
                }
            }
        }
    }

    fun extendGreenTime(seconds: Int) {
        _uiState.update { state ->
            if (state.currentPhase != TrafficPhase.ALL_RED) {
                state.copy(timeRemaining = state.timeRemaining + seconds)
            } else state
        }
    }

    fun manualControl(phase: TrafficPhase) {
        signalTimerJob?.cancel()
        _uiState.update {
            it.copy(
                currentPhase = phase,
                timeRemaining = 60,
                originalGreenTime = 60
            )
        }
        startSignalTimer()
    }

    fun resetToAutoMode() {
        _uiState.update {
            it.copy(
                currentPhase = TrafficPhase.NS_GREEN,
                timeRemaining = 5,
                originalGreenTime = 5
            )
        }
        startSignalTimer()
    }

    fun enableEmergencyOverride() {
        signalTimerJob?.cancel()
        _uiState.update {
            it.copy(
                isEmergency = it.isEmergency.not()
            )
        }
        if (_uiState.value.isEmergency) {
            makeAllRedSignal()
        } else {
            resetToAutoMode()
        }
    }

    fun makeAllRedSignal() {
        signalTimerJob?.cancel()
        _uiState.update {
            it.copy(
                currentPhase = TrafficPhase.ALL_RED,
                timeRemaining = 60
            )
        }
        startSignalTimer()
    }

    private fun computeAverage(phase: TrafficPhase): Double {
        val durations = greenDurations[phase] ?: return 0.0
        return if (durations.isNotEmpty()) durations.average() else 0.0
    }
}

data class MonitoringUiState(
    val isEmergency: Boolean = false,
    val isLoading: Boolean = true,
    val timeRemaining: Int = 5,
    val originalGreenTime: Int = 5,
    val currentPhase: TrafficPhase = TrafficPhase.NS_GREEN,
    val liveSignal: ModelLiveSignal? = null,
    val trafficPhase: TrafficPhase = TrafficPhase.NS_GREEN,
    val averageNSGreen: Double = 0.0,
    val averageEWGreen: Double = 0.0,
    val error: String? = null
)

//data class MonitoringUiState(
//    val isEmergency: Boolean = false,
//    val isLoading: Boolean = true,
//    val timeRemaining: Int = 5,
//    val currentPhase: TrafficPhase = TrafficPhase.NS_GREEN,
//    val liveSignal: ModelLiveSignal? = null,
//    val trafficPhase: TrafficPhase = TrafficPhase.NS_GREEN,
//    val error: String? = null
//)