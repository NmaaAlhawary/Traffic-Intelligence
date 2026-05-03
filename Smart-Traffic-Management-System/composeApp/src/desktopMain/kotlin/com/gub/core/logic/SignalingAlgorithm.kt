package com.gub.core.logic

import com.gub.features.monitoring.presentation.components.TrafficPhase
import com.gub.features.monitoring.viewModel.MonitoringUiState

object SignalingAlgorithm {

    fun getNextPhaseBasedOnData(state: MonitoringUiState): Pair<TrafficPhase, Int> {
        val nextGreenPhase = when (state.currentPhase) {
            TrafficPhase.NORTH_GREEN -> TrafficPhase.EAST_GREEN
            TrafficPhase.EAST_GREEN -> TrafficPhase.SOUTH_GREEN
            TrafficPhase.SOUTH_GREEN -> TrafficPhase.WEST_GREEN
            TrafficPhase.WEST_GREEN -> TrafficPhase.NORTH_GREEN
            TrafficPhase.ALL_RED -> TrafficPhase.NORTH_GREEN
        }

        return nextGreenPhase to state.greenPhaseDurationSeconds
    }
}
