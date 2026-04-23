package com.gub.features.dashboard.presentation.overview.components

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.gub.features.dashboard.domain.extension.*
import com.gub.features.dashboard.presentation.screens.components.SystemOverviewMetric
import com.gub.models.dashboard.overview.ModelSystemOverview

@Composable
fun SystemOverviewCard(systemOverview: ModelSystemOverview) {
    Card(
        modifier = Modifier.fillMaxWidth(),
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
                    text = "System Overview",
                    color = MaterialTheme.colorScheme.onSurface,
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold
                )

                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Dashboard,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = "16:28:43",
                        color = Color.Gray,
                        fontSize = 12.sp
                    )
                }
            }

            Spacer(modifier = Modifier.height(20.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                SystemOverviewMetric(
                    title = "System Health",
                    value = systemOverview.systemHealthValue(),
                    subtitle = systemOverview.systemHealthSubText(),
                    color = Color(0xFF4CAF50),
                    icon = Icons.Default.CheckCircle
                )

                SystemOverviewMetric(
                    title = "Weather Condition",
                    value = systemOverview.weatherTempValue(),
                    subtitle = systemOverview.weatherTempSubText(),
                    color = Color(0xFF4CAF50),
                    icon = Icons.Default.Traffic
                )

                SystemOverviewMetric(
                    title = "AI Response Time",
                    value = systemOverview.aiResponseTimeValue(),
                    subtitle = systemOverview.aiResponseTimeSubText(),
                    color = Color(0xFF4CAF50),
                    icon = Icons.Default.Psychology
                )

                SystemOverviewMetric(
                    title = "Current Flow",
                    value = systemOverview.currentFlowValue(),
                    subtitle = systemOverview.currentFlowSubText(),
                    color = Color(0xFF2196F3),
                    icon = Icons.Default.DirectionsCar
                )

                SystemOverviewMetric(
                    title = "Avg Wait Time",
                    value = systemOverview.avgWaitTimeValue(),
                    subtitle = systemOverview.avgWaitTimeSubText(),
                    color = Color(0xFF4CAF50),
                    icon = Icons.Default.Timer
                )
            }
        }
    }
}
