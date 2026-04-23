package com.gub.features.dashboard.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.gub.core.domain.Response
import com.gub.domain.models.dashboard.ModelLiveTraffic
import com.gub.features.dashboard.di.DashboardModule
import com.gub.features.dashboard.domain.usecase.GetSystemOverviewUseCase
import com.gub.features.dashboard.domain.usecase.UpdateSystemMetricsUseCase
import com.gub.models.dashboard.overview.ModelSystemOverview
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch

class ViewModelDashboard(
    private val getSystemOverviewUseCase: GetSystemOverviewUseCase = DashboardModule.getSystemOverviewUseCase,
    private val updateSystemMetricsUseCase: UpdateSystemMetricsUseCase = DashboardModule.updateSystemMetricsUseCase
) : ViewModel() {

    private val _uiState = MutableStateFlow(SystemOverviewUiState())
    val uiState: StateFlow<SystemOverviewUiState> = _uiState.asStateFlow()

    init {
        observeSystemOverview()
        liveTrafficMatrics()
    }

    private fun observeSystemOverview() {
        viewModelScope.launch {
            getSystemOverviewUseCase.getStream()
                .catch { exception ->
                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        error = exception.message
                    )
                }
                .collect { overview ->
                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        systemOverview = overview,
                        error = null
                    )
                }
        }
    }

    private fun liveTrafficMatrics() = viewModelScope.launch {
        when(val res = updateSystemMetricsUseCase()) {
            is Response.Error -> {}
            Response.Loading -> {}
            is Response.Success -> {
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    liveTrafficMatrics = res.data,
                    error = null
                )
            }
        }
    }
}

data class SystemOverviewUiState(
    val isLoading: Boolean = true,
    val liveTrafficMatrics: ModelLiveTraffic = ModelLiveTraffic(),
    val systemOverview: ModelSystemOverview = ModelSystemOverview(),
    val error: String? = null
)