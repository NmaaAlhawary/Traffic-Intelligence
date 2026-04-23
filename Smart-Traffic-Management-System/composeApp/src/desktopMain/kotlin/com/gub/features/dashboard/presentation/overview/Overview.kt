package com.gub.features.dashboard.presentation.overview

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.TrendingDown
import androidx.compose.material.icons.automirrored.filled.TrendingUp
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.gub.domain.models.dashboard.ModelLiveTraffic
import com.gub.features.dashboard.presentation.overview.components.AIStatusCard
import com.gub.features.dashboard.presentation.overview.components.SystemOverviewCard
import com.gub.features.dashboard.viewmodel.ViewModelDashboard
import dev.chrisbanes.haze.HazeState
import dev.chrisbanes.haze.hazeSource

@Composable
fun Overview(viewModelDashboard: ViewModelDashboard, hazeState: HazeState, top: Dp) {

    val systemOverview = viewModelDashboard.uiState.collectAsState()

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .hazeSource(hazeState),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        contentPadding = PaddingValues(top = top + 24.dp, start = 24.dp, end = 24.dp, bottom = 24.dp)
    ) {

        item { SystemOverviewCard(systemOverview.value.systemOverview) }

        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                LiveTrafficMetricsCard(
                    modifier = Modifier.weight(1f),
                    systemOverview.value.liveTrafficMatrics
                )
                AIStatusCard(modifier = Modifier.weight(1f))
            }
        }

//        item {
//            Row(
//                modifier = Modifier.fillMaxWidth(),
//                horizontalArrangement = Arrangement.spacedBy(16.dp)
//            ) {
//                RecentAlertsCard(modifier = Modifier.weight(1f))
//                TopPerformingIntersectionsCard(modifier = Modifier.weight(1f))
//            }
//        }
    }
}

@Composable
fun LiveTrafficMetricsCard(modifier: Modifier = Modifier, liveTrafficMatrics: ModelLiveTraffic) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    "Live Traffic Metrics",
                    color = MaterialTheme.colorScheme.onSurface,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )

                Card(
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primary)
                ) {
                    Text(
                        "REAL-TIME",
                        color = MaterialTheme.colorScheme.onPrimary,
                        fontSize = 8.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            LiveMetricItem(
                label = "Total Vehicle Count",
                value = (liveTrafficMatrics.vehicle.count * 20).toString().take(4),
                change = liveTrafficMatrics.vehicle.difference.toString().take(4),
                positive = liveTrafficMatrics.vehicle.upWards
            )
//            LiveMetricItem("Average Speed", "29.4 mph", "+1.2 mph improvement", true)
            LiveMetricItem(
                label = "Congestion Index",
                value = liveTrafficMatrics.congestion.count.toString().take(4),
                change = liveTrafficMatrics.congestion.difference.toString().take(4),
                positive = liveTrafficMatrics.congestion.upWards
            )

            print("DATA - > ${liveTrafficMatrics}")

//            LiveMetricItem("Signal Efficiency", "95.3%", "+2.1% optimization", true)

            Spacer(modifier = Modifier.height(12.dp))

            Card(
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primary.copy(0.1F))
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(12.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        "Next peak prediction:",
                        color = Color.Gray,
                        fontSize = 11.sp
                    )
                    Text(
                        "17:15 (High volume expected)",
                        color = Color(0xFF2196F3),
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Medium
                    )
                }
            }
        }
    }
}

@Composable
fun LiveMetricItem(label: String, value: String, change: String, positive: Boolean) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column {
            Text(
                label,
                color = Color.Gray,
                fontSize = 11.sp
            )
            Text(
                value,
                color = MaterialTheme.colorScheme.onSurface,
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold
            )
        }

        Column(horizontalAlignment = Alignment.End) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    if (positive) Icons.AutoMirrored.Filled.TrendingUp else Icons.AutoMirrored.Filled.TrendingDown,
                    contentDescription = null,
                    tint = if (positive) Color(0xFF4CAF50) else Color(0xFFF44336),
                    modifier = Modifier.size(12.dp)
                )
                Spacer(modifier = Modifier.width(4.dp))
            }
            Text(
                change,
                color = if (positive) Color(0xFF4CAF50) else Color(0xFFF44336),
                fontSize = 9.sp,
                textAlign = TextAlign.End
            )
        }
    }
}