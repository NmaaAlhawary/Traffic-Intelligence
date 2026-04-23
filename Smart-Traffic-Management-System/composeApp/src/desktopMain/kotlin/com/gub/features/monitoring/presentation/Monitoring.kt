package com.gub.features.monitoring.presentation

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.gub.features.dashboard.viewmodel.SystemOverviewUiState
import com.gub.features.dashboard.viewmodel.ViewModelDashboard
import com.gub.features.monitoring.data.network.LiveFeedManager
import com.gub.features.monitoring.presentation.components.IntersectionOverviewCard
import com.gub.features.monitoring.presentation.components.LiveCameraFeedCard
import com.gub.features.monitoring.presentation.components.TopBarMonitoring
import com.gub.features.monitoring.presentation.components.TrafficPhase
import com.gub.features.monitoring.viewModel.MonitoringUiState
import com.gub.features.monitoring.viewModel.ViewModelMonitoring
import com.gub.utils.UiCalculations.toDp
import dev.chrisbanes.haze.hazeSource
import dev.chrisbanes.haze.rememberHazeState

@Composable
fun Monitoring(viewModelMonitoring: ViewModelMonitoring) {
    var selectedIntersection by remember { mutableStateOf(0) }
    var isLiveView by remember { mutableStateOf(true) }
    var isCameraExpanded by remember { mutableStateOf(false) }

    val hazeState = rememberHazeState()
    var height by remember { mutableStateOf(0) }

    val intersectionName = remember { mutableStateOf("Mirpur 10") }

    val viewModelDashboard = remember { ViewModelDashboard() }

    val liveFeedManager = remember { LiveFeedManager() }
    var currentFrame by remember { mutableStateOf<ImageBitmap?>(null) }
    var vehicleCount by remember { mutableStateOf("{}") }
    var frameError by remember { mutableStateOf<String?>(null) }

    val uiState = viewModelMonitoring.uiState.collectAsState()

    val dashboard = viewModelDashboard.uiState.collectAsState()

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .hazeSource(hazeState),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        contentPadding = PaddingValues(top = height.toDp() + 24.dp, start = 24.dp, end = 24.dp, bottom = 24.dp)
    ) {

        if (isCameraExpanded) {
            item {
                LiveCameraFeedCard(
                    modifier = Modifier
                        .fillMaxWidth()
                        .animateContentSize(
                            animationSpec = tween(300)
                        ),
                    isExpanded = true,
                    currentFrame = currentFrame,
                    vehicleCount = vehicleCount,
                    frameError = frameError,
                    onToggleExpand = { isCameraExpanded = !isCameraExpanded }
                )
            }
        } else {
            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    IntersectionOverviewCard(
                        modifier = Modifier.weight(2f),
                        viewModelMonitoring = viewModelMonitoring,
                        vehicleCount = vehicleCount,
                        intersectionName = intersectionName
                    )

                    TrafficControlPanelCard(
                        modifier = Modifier.weight(1f),
                        viewModelMonitoring = viewModelMonitoring
                    )
                }
            }

            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    IntersectionSelectorCard(
                        selectedIntersection = selectedIntersection,
                        onIntersectionSelected = { selectedIntersection = it },
                        modifier = Modifier.weight(1f)
                    )
                    LiveCameraFeedCard(
                        modifier = Modifier
                            .weight(1f)
                            .animateContentSize(
                                animationSpec = tween(300)
                            ),
                        currentFrame = currentFrame,
                        vehicleCount = vehicleCount,
                        frameError = frameError,
                        onToggleExpand = { isCameraExpanded = !isCameraExpanded },
                    )
                }
            }

            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    VehicleCountCard(
                        modifier = Modifier.weight(1f),
                        uiState = uiState.value
                    )
                    SignalTimingCard(
                        modifier = Modifier.weight(1f),
                        uiState = uiState.value
                    )
                    EnvironmentalDataCard(
                        modifier = Modifier.weight(1f),
                        dashboard = dashboard.value
                    )
                }
            }
        }
    }

    TopBarMonitoring(
        hazeState,
        topBarHeight = { height = it }
    )

    LaunchedEffect(Unit) {
        try {
            liveFeedManager.streamVideoAndProcess("1.mp4") { frame, count ->
                currentFrame = frame
                vehicleCount = count
            }
        } catch (e: Exception) {
            frameError = e.message
        }
    }
}

@Composable
fun TrafficControlPanelCard(modifier: Modifier = Modifier, viewModelMonitoring: ViewModelMonitoring) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        shape = RoundedCornerShape(12.dp)
    ) {

        val uiState = viewModelMonitoring.uiState.collectAsState()
        var position by remember { mutableStateOf(0) }

        Column {
            AnimatedContent(
                targetState = position
            ) {
                when(it) {
                    0 -> {
                        // Main Control Panel
                        Column(
                            modifier = Modifier.padding(20.dp)
                        ) {
                            Text(
                                "Traffic Control",
                                color = MaterialTheme.colorScheme.onSurface,
                                fontSize = 18.sp,
                                fontWeight = FontWeight.Bold
                            )
                            Spacer(modifier = Modifier.height(16.dp))

                            ControlButton(
                                text = "Make All Override",
                                icon = Icons.Default.Warning,
                                color = Color(0xFFF44336),
                                description = "Clear all lanes immediately",
                                onClick = { position = 1 }
                            )

                            Spacer(modifier = Modifier.height(12.dp))

                            ControlButton(
                                text = "Extend Green Time",
                                icon = Icons.Default.Timer,
                                color = Color(0xFF4CAF50),
                                description = "Add 30 seconds to current phase",
                                onClick = { position = 2 }
                            )

                            Spacer(modifier = Modifier.height(12.dp))

                            ControlButton(
                                text = "Manual Control",
                                icon = Icons.Default.TouchApp,
                                color = MaterialTheme.colorScheme.primary,
                                description = "Take direct control of signals",
                                onClick = { position = 3 }
                            )

                            Spacer(modifier = Modifier.height(12.dp))

                            ControlButton(
                                text = "Reset to Auto",
                                icon = Icons.Default.Autorenew,
                                color = Color(0xFFFF9800),
                                description = "Return to AI optimization",
                                onClick = { position = 4 }
                            )
                        }
                    }
                    1 -> {
                        // Override Control Page
                        Column(
                            modifier = Modifier.padding(20.dp)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                IconButton(
                                    onClick = { position = 0 }
                                ) {
                                    Icon(
                                        Icons.Default.ArrowBack,
                                        contentDescription = "Back",
                                        tint = MaterialTheme.colorScheme.onSurface
                                    )
                                }
                                Text(
                                    "Override Control",
                                    color = MaterialTheme.colorScheme.onSurface,
                                    fontSize = 18.sp,
                                    fontWeight = FontWeight.Bold
                                )
                                Spacer(modifier = Modifier.width(48.dp))
                            }

                            Spacer(modifier = Modifier.height(16.dp))

                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                colors = CardDefaults.cardColors(
                                    containerColor = Color(0xFFF44336).copy(alpha = 0.1f)
                                )
                            ) {
                                Column(
                                    modifier = Modifier.padding(16.dp)
                                ) {
                                    Icon(
                                        Icons.Default.Warning,
                                        contentDescription = null,
                                        tint = Color(0xFFF44336),
                                        modifier = Modifier.size(24.dp)
                                    )
                                    Spacer(modifier = Modifier.height(8.dp))
                                    Text(
                                        "Emergency Override",
                                        fontWeight = FontWeight.Bold,
                                        color = Color(0xFFF44336)
                                    )
                                    Text(
                                        "This will immediately clear all traffic lanes. Use only in emergency situations.",
                                        color = MaterialTheme.colorScheme.onSurface,
                                        fontSize = 14.sp,
                                        modifier = Modifier.padding(top = 4.dp)
                                    )
                                }
                            }

                            Spacer(modifier = Modifier.height(20.dp))

                            Button(
                                onClick = {
                                    viewModelMonitoring.enableEmergencyOverride()
                                },
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = Color(0xFFF44336)
                                ),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text(
                                    if (uiState.value.isEmergency)
                                        "DEACTIVATE OVERRIDE" else
                                            "ACTIVATE OVERRIDE"
                                )
                            }
                        }
                    }
                    2 -> {
                        // Extend Green Time Page
                        Column(
                            modifier = Modifier.padding(20.dp)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                IconButton(
                                    onClick = { position = 0 }
                                ) {
                                    Icon(
                                        Icons.Default.ArrowBack,
                                        contentDescription = "Back",
                                        tint = MaterialTheme.colorScheme.onSurface
                                    )
                                }
                                Text(
                                    "Extend Green Time",
                                    color = MaterialTheme.colorScheme.onSurface,
                                    fontSize = 18.sp,
                                    fontWeight = FontWeight.Bold
                                )
                                Spacer(modifier = Modifier.width(48.dp))
                            }

                            Spacer(modifier = Modifier.height(16.dp))

                            var selectedTime by remember { mutableStateOf(30) }
                            val timeOptions = listOf(15, 30, 45, 60)

                            Text(
                                "Select extension time:",
                                color = MaterialTheme.colorScheme.onSurface,
                                fontWeight = FontWeight.Medium
                            )

                            Spacer(modifier = Modifier.height(12.dp))

                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceEvenly
                            ) {
                                timeOptions.forEach { time ->
                                    FilterChip(
                                        selected = selectedTime == time,
                                        onClick = { selectedTime = time },
                                        label = { Text("${time}s") }
                                    )
                                }
                            }

                            Spacer(modifier = Modifier.height(20.dp))

                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                colors = CardDefaults.cardColors(
                                    containerColor = Color(0xFF4CAF50).copy(alpha = 0.1f)
                                )
                            ) {
                                Column(
                                    modifier = Modifier.padding(16.dp)
                                ) {
                                    Text(
                                        "Current Phase: ${uiState.value.currentPhase}",
                                        fontWeight = FontWeight.Bold,
                                        color = MaterialTheme.colorScheme.onSurface
                                    )
                                    Text(
                                        "Time remaining: ${uiState.value.timeRemaining} seconds",
                                        color = MaterialTheme.colorScheme.onSurface,
                                        fontSize = 14.sp
                                    )
                                }
                            }

                            Spacer(modifier = Modifier.height(20.dp))

                            Button(
                                onClick = {
                                    viewModelMonitoring.extendGreenTime(selectedTime)
                                },
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = Color(0xFF4CAF50)
                                ),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text("EXTEND BY ${selectedTime}S")
                            }
                        }
                    }
                    3 -> {
                        // Manual Control Page
                        Column(
                            modifier = Modifier.padding(20.dp)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                IconButton(
                                    onClick = { position = 0 }
                                ) {
                                    Icon(
                                        Icons.Default.ArrowBack,
                                        contentDescription = "Back",
                                        tint = MaterialTheme.colorScheme.onSurface
                                    )
                                }
                                Text(
                                    "Manual Control",
                                    color = MaterialTheme.colorScheme.onSurface,
                                    fontSize = 18.sp,
                                    fontWeight = FontWeight.Bold
                                )
                                Spacer(modifier = Modifier.width(48.dp))
                            }

                            Spacer(modifier = Modifier.height(16.dp))

                            Text(
                                "Select signal direction:",
                                color = MaterialTheme.colorScheme.onSurface,
                                fontWeight = FontWeight.Medium
                            )

                            Spacer(modifier = Modifier.height(16.dp))

                            ManualControlButton(
                                text = "North-South Green",
                                isActive = uiState.value.currentPhase == TrafficPhase.NS_GREEN,
                                onClick = {
                                    viewModelMonitoring.manualControl(
                                        TrafficPhase.NS_GREEN
                                    )
                                }
                            )

                            Spacer(modifier = Modifier.height(12.dp))

                            ManualControlButton(
                                text = "East-West Green",
                                isActive = uiState.value.currentPhase == TrafficPhase.EW_GREEN,
                                onClick = {
                                    viewModelMonitoring.manualControl(
                                        TrafficPhase.EW_GREEN
                                    )
                                }
                            )

                            Spacer(modifier = Modifier.height(12.dp))

                            ManualControlButton(
                                text = "All Red (Stop)",
                                isActive = uiState.value.currentPhase == TrafficPhase.ALL_RED,
                                onClick = {
                                    viewModelMonitoring.manualControl(
                                        TrafficPhase.ALL_RED
                                    )
                                }
                            )

                            Spacer(modifier = Modifier.height(20.dp))

                            OutlinedButton(
                                onClick = { position = 0 },
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text("EXIT MANUAL MODE")
                            }
                        }
                    }
                    4 -> {
                        // Reset to Auto Page
                        Column(
                            modifier = Modifier.padding(20.dp)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                IconButton(
                                    onClick = { position = 0 }
                                ) {
                                    Icon(
                                        Icons.Default.ArrowBack,
                                        contentDescription = "Back",
                                        tint = MaterialTheme.colorScheme.onSurface
                                    )
                                }
                                Text(
                                    "Reset to Auto",
                                    color = MaterialTheme.colorScheme.onSurface,
                                    fontSize = 18.sp,
                                    fontWeight = FontWeight.Bold
                                )
                                Spacer(modifier = Modifier.width(48.dp))
                            }

                            Spacer(modifier = Modifier.height(16.dp))

                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                colors = CardDefaults.cardColors(
                                    containerColor = Color(0xFFFF9800).copy(alpha = 0.1f)
                                )
                            ) {
                                Column(
                                    modifier = Modifier.padding(16.dp)
                                ) {
                                    Icon(
                                        Icons.Default.Autorenew,
                                        contentDescription = null,
                                        tint = Color(0xFFFF9800),
                                        modifier = Modifier.size(24.dp)
                                    )
                                    Spacer(modifier = Modifier.height(8.dp))
                                    Text(
                                        "Return to AI Control",
                                        fontWeight = FontWeight.Bold,
                                        color = Color(0xFFFF9800)
                                    )
                                    Text(
                                        "This will return traffic control to AI optimization based on real-time traffic patterns.",
                                        color = MaterialTheme.colorScheme.onSurface,
                                        fontSize = 14.sp,
                                        modifier = Modifier.padding(top = 4.dp)
                                    )
                                }
                            }

                            Spacer(modifier = Modifier.height(16.dp))

                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                colors = CardDefaults.cardColors(
                                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                                )
                            ) {
                                Column(
                                    modifier = Modifier.padding(16.dp)
                                ) {
                                    Text(
                                        "Current Status:",
                                        fontWeight = FontWeight.Bold,
                                        color = MaterialTheme.colorScheme.onSurface
                                    )
                                    Text(
                                        "Manual override active",
                                        color = MaterialTheme.colorScheme.onSurface,
                                        fontSize = 14.sp
                                    )
                                    Text(
                                        "Duration: 5 minutes 23 seconds",
                                        color = MaterialTheme.colorScheme.onSurface,
                                        fontSize = 14.sp
                                    )
                                }
                            }

                            Spacer(modifier = Modifier.height(20.dp))

                            Button(
                                onClick = { /* Handle reset */ },
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = Color(0xFFFF9800)
                                ),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text("RESET TO AUTO")
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun ControlButton(
    text: String,
    icon: ImageVector,
    color: Color,
    description: String,
    onClick: () -> Unit = {}
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() },
        colors = CardDefaults.cardColors(
            containerColor = color.copy(alpha = 0.1f)
        ),
        shape = RoundedCornerShape(8.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                icon,
                contentDescription = null,
                tint = color,
                modifier = Modifier.size(24.dp)
            )
            Spacer(modifier = Modifier.width(16.dp))
            Column(
                modifier = Modifier.weight(1f)
            ) {
                Text(
                    text,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    description,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
                    fontSize = 12.sp
                )
            }
            Icon(
                Icons.Default.ChevronRight,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
            )
        }
    }
}

@Composable
fun ManualControlButton(
    text: String,
    isActive: Boolean,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() },
        colors = CardDefaults.cardColors(
            containerColor = if (isActive)
                MaterialTheme.colorScheme.primary.copy(alpha = 0.1f)
            else
                MaterialTheme.colorScheme.surfaceVariant
        ),
        shape = RoundedCornerShape(8.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            RadioButton(
                selected = isActive,
                onClick = onClick
            )
            Spacer(modifier = Modifier.width(12.dp))
            Text(
                text,
                color = MaterialTheme.colorScheme.onSurface,
                fontWeight = if (isActive) FontWeight.Bold else FontWeight.Normal
            )
        }
    }
}

@Composable
fun ControlButton(text: String, icon: ImageVector, color: Color, description: String) {
    OutlinedButton(
        onClick = { },
        modifier = Modifier.fillMaxWidth(),
        colors = ButtonDefaults.outlinedButtonColors(contentColor = color),
        border = BorderStroke(1.dp, color.copy(alpha = 0.5f))
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    icon,
                    contentDescription = null,
                    modifier = Modifier.size(12.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(text, fontSize = 12.sp, fontWeight = FontWeight.Medium)
            }
            Text(
                description,
                fontSize = 9.sp,
                color = Color.Gray
            )
        }
    }
}

@Composable
fun IntersectionSelectorCard(
    selectedIntersection: Int,
    onIntersectionSelected: (Int) -> Unit,
    modifier: Modifier = Modifier
) {
    val intersections = listOf(
        "Begum Rokeya Ave & Mirpur Rd",
        "Gulshan 1 & Gulshan 2",
        "Bijoy Sarani & Pragati Sarani",
        "Shahbagh & Nilkhet",
    )

    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp)
        ) {
            Text(
                "Select Intersection",
                color = MaterialTheme.colorScheme.onSurface,
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(16.dp))

            LazyColumn(
                verticalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.height(160.dp)
            ) {
                items(intersections.size) { index ->
                    IntersectionItem(
                        name = intersections[index],
                        isSelected = selectedIntersection == index,
                        onClick = { onIntersectionSelected(index) },
                        status = when (index % 3) {
                            0 -> "Normal"
                            1 -> "Congested"
                            else -> "Alert"
                        }
                    )
                }
            }
        }
    }
}

@Composable
fun IntersectionItem(
    name: String,
    isSelected: Boolean,
    onClick: () -> Unit,
    status: String
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() },
        colors = CardDefaults.cardColors(
            containerColor = if (isSelected) Color(0xFF2E7D32) else MaterialTheme.colorScheme.surfaceVariant
        ),
        shape = RoundedCornerShape(6.dp)
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    name,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Medium
                )
                Text(
                    status,
                    color = when (status) {
                        "Normal" -> Color(0xFF4CAF50)
                        "Congested" -> Color(0xFFFF9800)
                        else -> Color(0xFFF44336)
                    },
                    fontSize = 10.sp
                )
            }

            if (isSelected) {
                Icon(
                    Icons.Default.CheckCircle,
                    contentDescription = null,
                    tint = Color(0xFF4CAF50),
                    modifier = Modifier.size(16.dp)
                )
            }
        }
    }
}

@Composable
fun VehicleCountCard(modifier: Modifier = Modifier, uiState: MonitoringUiState) {

    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp)
        ) {
            Text(
                "Vehicle Count",
                color = MaterialTheme.colorScheme.onSurface,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(12.dp))

            VehicleCountItem("Cars", 156, Color(0xFF2196F3))
            VehicleCountItem("Trucks", 23, Color(0xFFFF9800))
            VehicleCountItem("Buses", 8, Color(0xFF4CAF50))
            VehicleCountItem("Motorcycles", 12, Color(0xFF9C27B0))
        }
    }
}

@Composable
fun VehicleCountItem(type: String, count: Int, color: Color) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .background(color, CircleShape)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(type, color = Color.Gray, fontSize = 12.sp)
        }
        Text(
            count.toString(),
            color = MaterialTheme.colorScheme.onSurface,
            fontSize = 14.sp,
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
fun SignalTimingCard(modifier: Modifier = Modifier, uiState: MonitoringUiState) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp)
        ) {
            Text(
                "Signal Timing",
                color = MaterialTheme.colorScheme.onSurface,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(12.dp))

            SignalPhase("North-South", uiState.averageNSGreen.toInt(), uiState.timeRemaining, TrafficLightState.GREEN)
            SignalPhase("East-West", uiState.averageEWGreen.toInt(), uiState.timeRemaining, TrafficLightState.RED)
//            SignalPhase("Left Turn", 15, 0, TrafficLightState.RED)
        }
    }
}

@Composable
fun SignalPhase(direction: String, totalTime: Int, remaining: Int, state: TrafficLightState) {
    Column(modifier = Modifier.padding(vertical = 4.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .background(state.color, CircleShape)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(direction, color = Color.Gray, fontSize = 12.sp)
            }
            Text(
                "${remaining}s",
                color = MaterialTheme.colorScheme.onSurface,
                fontSize = 12.sp,
                fontWeight = FontWeight.Medium
            )
        }

        Spacer(modifier = Modifier.height(4.dp))

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(4.dp)
                .background(Color(0xFF30363D), RoundedCornerShape(2.dp))
        ) {
            Box(
                modifier = Modifier
                    .fillMaxHeight()
                    .fillMaxWidth(if (totalTime > 0) remaining.toFloat() / totalTime else 0f)
                    .background(state.color, RoundedCornerShape(2.dp))
            )
        }
    }
}

@Composable
fun EnvironmentalDataCard(modifier: Modifier = Modifier, dashboard: SystemOverviewUiState) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp)
        ) {
            Text(
                "Environmental",
                color = MaterialTheme.colorScheme.onSurface,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(12.dp))

            EnvironmentalItem("Visibility", dashboard.systemOverview.weather.visibility.toString(), Color(0xFF4CAF50))
            EnvironmentalItem("Weather", "Sunny", Color(0xFF4CAF50))
            EnvironmentalItem("Temperature", dashboard.systemOverview.weather.temp.toString() + " C", Color(0xFF4CAF50))
            EnvironmentalItem("Wind", dashboard.systemOverview.weather.wind.toString() + " mph", Color(0xFF4CAF50))
        }
    }
}

@Composable
fun EnvironmentalItem(label: String, value: String, color: Color) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(label, color = Color.Gray, fontSize = 12.sp)
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(6.dp)
                    .background(color, CircleShape)
            )
            Spacer(modifier = Modifier.width(6.dp))
            Text(
                value,
                color = MaterialTheme.colorScheme.onSurface,
                fontSize = 12.sp,
                fontWeight = FontWeight.Medium
            )
        }
    }
}

// Additional data classes for the monitoring screen
enum class TrafficLightState(val color: Color) {
    RED(Color(0xFFF44336)),
    YELLOW(Color(0xFFFF9800)),
    GREEN(Color(0xFF4CAF50))
}