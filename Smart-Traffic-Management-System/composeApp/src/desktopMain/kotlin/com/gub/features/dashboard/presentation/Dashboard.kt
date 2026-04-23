package com.gub.features.dashboard.presentation

import androidx.compose.animation.AnimatedContent
import androidx.compose.runtime.*
import com.gub.features.dashboard.presentation.components.TopBarDashboard
import com.gub.features.dashboard.presentation.insights.UserInsightsDashboardCard
import com.gub.features.dashboard.presentation.liveStatus.LiveStatus
import com.gub.features.dashboard.presentation.overview.Overview
import com.gub.features.dashboard.presentation.quickAction.QuickActionsGrid
import com.gub.features.dashboard.viewmodel.ViewModelDashboard
import com.gub.utils.UiCalculations.toDp
import dev.chrisbanes.haze.rememberHazeState

@Composable
fun Dashboard() {

    val hazeState = rememberHazeState()
    var height by remember { mutableStateOf(0) }
    var selectedTab by remember { mutableStateOf(0) }

    val viewModelDashboard by remember { mutableStateOf(ViewModelDashboard()) }

    AnimatedContent(
        targetState = selectedTab
    ) {
        when(it) {
            0 -> Overview(viewModelDashboard, hazeState, height.toDp())
            1 -> LiveStatus(viewModelDashboard, hazeState, height.toDp())
            2 -> QuickActionsGrid(viewModelDashboard, hazeState, height.toDp())
            3 -> UserInsightsDashboardCard(viewModelDashboard, hazeState, height.toDp())
        }
    }

    TopBarDashboard(
        hazeState,
        selectedTab,
        onSelect = { selectedTab = it },
        topBarHeight = { height = it }
    )
}