package com.gub.core.logic

import com.gub.features.monitoring.presentation.components.TrafficPhase
import com.gub.features.monitoring.viewModel.MonitoringUiState

object SignalingAlgorithm {

    private val alpha = 1.0
    private val beta = 0.5

    fun getNextPhaseBasedOnData(state: MonitoringUiState): Pair<TrafficPhase, Int> {
        val data = state.liveSignal ?: return fallbackPhase(state.currentPhase)

        // Step 1: Decide the direction to switch to (alternating)
        val nextGreenPhase = when (state.currentPhase) {
            TrafficPhase.NS_GREEN, TrafficPhase.NS_YELLOW -> TrafficPhase.EW_GREEN
            TrafficPhase.EW_GREEN, TrafficPhase.EW_YELLOW -> TrafficPhase.NS_GREEN
            TrafficPhase.ALL_RED -> TrafficPhase.NS_GREEN
        }

        // Step 2: Dynamically calculate green duration based on vehicle count & wait time
        val greenDuration = when (nextGreenPhase) {
            TrafficPhase.NS_GREEN -> {
                val scoreNorth = calculateScore(data.north.vehicleCount, data.waitingTime)
                val scoreSouth = calculateScore(data.south.vehicleCount, data.waitingTime)
                (scoreNorth + scoreSouth).toInt().coerceIn(20, 60) // limit between 20-60 sec
            }
            TrafficPhase.EW_GREEN -> {
                val scoreEast = calculateScore(data.east.vehicleCount, data.waitingTime)
                val scoreWest = calculateScore(data.west.vehicleCount, data.waitingTime)
                (scoreEast + scoreWest).toInt().coerceIn(20, 60) // limit between 20-60 sec
            }
            else -> 45 // fallback
        }

        // Step 3: Transition logic with fixed yellow duration
        return when (state.currentPhase) {
            TrafficPhase.NS_GREEN -> TrafficPhase.NS_YELLOW to 4
            TrafficPhase.NS_YELLOW -> nextGreenPhase to greenDuration
            TrafficPhase.EW_GREEN -> TrafficPhase.EW_YELLOW to 4
            TrafficPhase.EW_YELLOW -> nextGreenPhase to greenDuration
            TrafficPhase.ALL_RED -> TrafficPhase.EW_YELLOW to 4
        }
    }

    private fun calculateScore(vehicleCount: Int, waitTime: Long): Double {
        return alpha * vehicleCount + beta * waitTime
    }

    private fun fallbackPhase(current: TrafficPhase): Pair<TrafficPhase, Int> {
        return when (current) {
            TrafficPhase.NS_GREEN -> TrafficPhase.NS_YELLOW to 4
            TrafficPhase.NS_YELLOW -> TrafficPhase.EW_GREEN to 35
            TrafficPhase.EW_GREEN -> TrafficPhase.EW_YELLOW to 4
            TrafficPhase.EW_YELLOW -> TrafficPhase.NS_GREEN to 45
            TrafficPhase.ALL_RED -> TrafficPhase.NS_GREEN to 45
        }
    }
}