package com.gub.features.monitoring.presentation.components

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.gub.domain.models.monitoring.ModelLiveSignal
import com.gub.domain.models.monitoring.ModelLiveSignal.Road.SignalType
import com.gub.features.monitoring.domain.model.VehicleCountJson
import com.gub.features.monitoring.presentation.TrafficLightState
import com.gub.features.monitoring.viewModel.MonitoringUiState
import com.gub.features.monitoring.viewModel.ViewModelMonitoring
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.StateFlow
import kotlinx.serialization.json.Json
import kotlin.random.Random

@Composable
fun IntersectionOverviewCard(
    modifier: Modifier = Modifier,
    viewModelMonitoring: ViewModelMonitoring,
    vehicleCount: String,
    intersectionName: MutableState<String>
) {

    var selectedView by remember { mutableStateOf("Live") }

    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(
            modifier = Modifier.padding(24.dp)
        ) {
            // Modern Header
            ModernHeader(intersectionName = intersectionName)

            Spacer(modifier = Modifier.height(20.dp))

            // View Selector
//            ViewSelector(selectedView) { selectedView = it }

            Spacer(modifier = Modifier.height(24.dp))

            // Main Intersection Display
            when (selectedView) {
                "Live" -> LiveIntersectionView(
                    viewModelMonitoring,
                    vehicleCount = vehicleCount,
                    intersectionName = intersectionName
                )
                "Stats" -> StatsView()
                "History" -> HistoryView()
            }

            Spacer(modifier = Modifier.height(20.dp))

            // Quick Metrics
            QuickMetrics(vehicleCount)
        }
    }
}

@Composable
private fun ModernHeader(intersectionName: MutableState<String>) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column {
            Row(
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(28.dp)
                        .background(
                            Brush.linearGradient(
                                colors = listOf(Color(0xFF00D4FF), Color(0xFF0099CC))
                            ),
                            RoundedCornerShape(8.dp)
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        Icons.Default.Traffic,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(16.dp)
                    )
                }
                Spacer(modifier = Modifier.width(16.dp))
                Column {
                    Text(
                        text = intersectionName.value,
                        color = MaterialTheme.colorScheme.onSurface,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold
                    )
//                    Text(
//                        "Times Square District • User: Alims-Repo",
//                        color = MaterialTheme.colorScheme.onSurface.copy(0.75F),
//                        fontSize = 11.sp
//                    )
                }
            }
        }

        StatusBadge()
    }
}

@Composable
private fun StatusBadge() {
    var pulseAlpha by remember { mutableStateOf(0.8f) }

    LaunchedEffect(Unit) {
        while (true) {
            pulseAlpha = 1f
            delay(600)
            pulseAlpha = 0.4f
            delay(600)
        }
    }

    Card(
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFF059669).copy(alpha = 0.2f)
        ),
        shape = RoundedCornerShape(16.dp),
        border = BorderStroke(1.dp, Color(0xFF059669).copy(alpha = 0.4f))
    ) {
        Row(
            modifier = Modifier.padding(12.dp, 8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .background(
                        Color(0xFF10B981).copy(alpha = pulseAlpha),
                        CircleShape
                    )
            )
            Spacer(modifier = Modifier.width(8.dp))

            Text(
                "ACTIVE",
                color = Color(0xFF10B981),
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 0.5.sp
            )
        }
    }
}

@Composable
private fun ViewSelector(selectedView: String, onViewChange: (String) -> Unit) {
    val views = listOf("Live", "Stats", "History")

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(
                color = MaterialTheme.colorScheme.surfaceVariant,
                shape = RoundedCornerShape(12.dp)
            )
            .padding(4.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        views.forEach { view ->
            val isSelected = view == selectedView

            Box(
                modifier = Modifier
                    .weight(1f)
                    .background(
                        if (isSelected) {
                            Brush.linearGradient(
                                colors = listOf(Color(0xFF3B82F6), Color(0xFF1D4ED8))
                            )
                        } else {
                            Brush.linearGradient(
                                colors = listOf(Color.Transparent, Color.Transparent)
                            )
                        },
                        RoundedCornerShape(8.dp)
                    )
                    .clickable { onViewChange(view) }
                    .padding(12.dp, 8.dp),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = view,
                    color = if (selectedView == view)
                        MaterialTheme.colorScheme.onPrimary
                    else MaterialTheme.colorScheme.onSurface,
                    fontSize = 12.sp,
                    fontWeight = if (isSelected) FontWeight.SemiBold else FontWeight.Medium
                )
            }
        }
    }
}

@Composable
private fun LiveIntersectionView(
    viewModelMonitoring: ViewModelMonitoring,
    vehicleCount: String,
    intersectionName: MutableState<String>
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(280.dp)
            .background(
                Brush.radialGradient(
                    colors = listOf(
                        Color(0xFF1E3A8A).copy(alpha = 0.15f),
                        Color.Transparent
                    ),
                    radius = 400f
                ),
                RoundedCornerShape(16.dp)
            )
    ) {
        // Modern Road Network
        ModernRoadNetwork()

        // Smart Traffic Lights
        SmartTrafficLights(viewModelMonitoring)

        // Dynamic Vehicles
//        DynamicVehicleFlow(
//            viewModelMonitoring,
//            vehicleCount = vehicleCount
//        )

        // Traffic Info Overlay
//        TrafficInfoOverlay()
    }
}

@Composable
private fun BoxScope.ModernRoadNetwork() {
    Canvas(modifier = Modifier.fillMaxSize()) {
        val centerX = size.width / 2
        val centerY = size.height / 2
        val roadWidth = 80.dp.toPx()
        val laneWidth = roadWidth / 3

        // Broadway (horizontal) - main thoroughfare
        drawRoundRect(
            color = Color(0xFF1E293B),
            topLeft = Offset(0f, centerY - roadWidth / 2),
            size = Size(size.width, roadWidth),
            cornerRadius = CornerRadius(8.dp.toPx())
        )

        // Lane dividers for Broadway
        for (i in 1..2) {
            drawLine(
                color = Color(0xFF374151),
                start = Offset(0f, centerY - roadWidth / 2 + i * laneWidth),
                end = Offset(size.width, centerY - roadWidth / 2 + i * laneWidth),
                strokeWidth = 2.dp.toPx(),
                pathEffect = PathEffect.dashPathEffect(floatArrayOf(10f, 10f))
            )
        }

        // 42nd Street (vertical)
        drawRoundRect(
            color = Color(0xFF1E293B),
            topLeft = Offset(centerX - roadWidth / 2, 0f),
            size = Size(roadWidth, size.height),
            cornerRadius = CornerRadius(8.dp.toPx())
        )

        // Lane dividers for 42nd Street
        for (i in 1..2) {
            drawLine(
                color = Color(0xFF374151),
                start = Offset(centerX - roadWidth / 2 + i * laneWidth, 0f),
                end = Offset(centerX - roadWidth / 2 + i * laneWidth, size.height),
                strokeWidth = 2.dp.toPx(),
                pathEffect = PathEffect.dashPathEffect(floatArrayOf(10f, 10f))
            )
        }

        // Intersection center with crosswalk patterns
        drawRoundRect(
            color = Color(0xFF374151),
            topLeft = Offset(centerX - roadWidth / 2, centerY - roadWidth / 2),
            size = Size(roadWidth, roadWidth),
            cornerRadius = CornerRadius(8.dp.toPx())
        )

        // Crosswalk stripes

        val stripeWidth = 4.dp.toPx()
        val spacing = 8.dp.toPx()
        val totalStripeWidth = stripeWidth + spacing

        for (i in 0..6) {
            val offset = i * totalStripeWidth

            // Horizontal crosswalk stripes (vertical lines)
            drawRect(
                color = Color.White.copy(alpha = 0.5f),
                topLeft = Offset(centerX - roadWidth / 2 + offset, centerY - roadWidth / 2),
                size = Size(stripeWidth, roadWidth)
            )

            // Vertical crosswalk stripes (horizontal lines)
            drawRect(
                color = Color.White.copy(alpha = 0.5f),
                topLeft = Offset(centerX - roadWidth / 2, centerY - roadWidth / 2 + offset),
                size = Size(roadWidth, stripeWidth)
            )
        }

//        for (i in 0..6) {
//            val stripeWidth = 4.dp.toPx()
//            val spacing = 8.dp.toPx()
//
//            // Horizontal crosswalks
//            drawRect(
//                color = Color.White.copy(alpha = 0.5f),
//                topLeft = Offset(centerX - roadWidth / 2 + i * spacing, centerY - roadWidth / 2),
//                size = Size(stripeWidth, roadWidth)
//            )
//
//            // Vertical crosswalks
//            drawRect(
//                color = Color.White.copy(alpha = 0.5f),
//                topLeft = Offset(centerX - roadWidth / 2, centerY - roadWidth / 2 + i * spacing),
//                size = Size(roadWidth, stripeWidth)
//            )
//        }
    }

    // Street name labels with modern styling
//    Card(
//        modifier = Modifier
//            .align(Alignment.CenterStart)
//            .padding(start = 20.dp),
//        colors = CardDefaults.cardColors(
//            containerColor = Color.Black.copy(alpha = 0.8f)
//        ),
//        shape = RoundedCornerShape(8.dp)
//    ) {
//        Text(
//            "BROADWAY",
//            color = Color(0xFF00D4FF),
//            fontSize = 11.sp,
//            fontWeight = FontWeight.Bold,
//            letterSpacing = 1.sp,
//            modifier = Modifier.padding(8.dp, 4.dp)
//        )
//    }

//    Card(
//        modifier = Modifier
//            .align(Alignment.TopCenter)
//            .padding(top = 20.dp),
//        colors = CardDefaults.cardColors(
//            containerColor = Color.Black.copy(alpha = 0.8f)
//        ),
//        shape = RoundedCornerShape(8.dp)
//    ) {
//        Text(
//            "42ND ST",
//            color = Color(0xFF00D4FF),
//            fontSize = 11.sp,
//            fontWeight = FontWeight.Bold,
//            letterSpacing = 1.sp,
//            modifier = Modifier.padding(8.dp, 4.dp)
//        )
//    }
}

@Composable
private fun SmartTrafficLights(viewModelMonitoring: ViewModelMonitoring) {

//    var currentPhase by remember { mutableStateOf(TrafficPhase.NS_GREEN) }
//    var timeRemaining by remember { mutableStateOf(45) }
//
//    LaunchedEffect(Unit) {
//        while (true) {
//            delay(1000)
//            timeRemaining--
//            if (timeRemaining <= 0) {
//                currentPhase = when (currentPhase) {
//                    TrafficPhase.NS_GREEN -> {
//                        timeRemaining = 4
//                        TrafficPhase.NS_YELLOW
//                    }
//                    TrafficPhase.NS_YELLOW -> {
//                        timeRemaining = 35
//                        TrafficPhase.EW_GREEN
//                    }
//                    TrafficPhase.EW_GREEN -> {
//                        timeRemaining = 4
//                        TrafficPhase.EW_YELLOW
//                    }
//                    TrafficPhase.EW_YELLOW -> {
//                        timeRemaining = 45
//                        TrafficPhase.NS_GREEN
//                    }
//                }
//            }
//        }
//    }

    val uiState by viewModelMonitoring.uiState.collectAsState()

    Box(modifier = Modifier.fillMaxSize()) {
        SmartSignalBox(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .offset(x = 60.dp, y = 0.dp),
            direction = "N",
            state = when (uiState.currentPhase) {
                TrafficPhase.NS_GREEN -> TrafficLightState.GREEN
                TrafficPhase.NS_YELLOW -> TrafficLightState.YELLOW
                else -> TrafficLightState.RED
            },
            timeRemaining = if (uiState.currentPhase == TrafficPhase.NS_GREEN || uiState.currentPhase == TrafficPhase.NS_YELLOW) uiState.timeRemaining else 0
        )

        // East Signal (for westbound traffic from Broadway)
        SmartSignalBox(
            modifier = Modifier
                .align(Alignment.CenterEnd)
                .offset(x = (-50).dp, y = 25.dp),
            direction = "E",
            state = when (uiState.currentPhase) {
                TrafficPhase.EW_GREEN -> TrafficLightState.GREEN
                TrafficPhase.EW_YELLOW -> TrafficLightState.YELLOW
                else -> TrafficLightState.RED
            },
            timeRemaining = if (uiState.currentPhase == TrafficPhase.EW_GREEN || uiState.currentPhase == TrafficPhase.EW_YELLOW) uiState.timeRemaining else 0
        )

        // South Signal (for northbound traffic from 42nd St)
        SmartSignalBox(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .offset(x = (-60).dp, y = (-0).dp),
            direction = "S",
            state = when (uiState.currentPhase) {
                TrafficPhase.NS_GREEN -> TrafficLightState.GREEN
                TrafficPhase.NS_YELLOW -> TrafficLightState.YELLOW
                else -> TrafficLightState.RED
            },
            timeRemaining = if (uiState.currentPhase == TrafficPhase.NS_GREEN || uiState.currentPhase == TrafficPhase.NS_YELLOW) uiState.timeRemaining else 0
        )

        // West Signal (for eastbound traffic from Broadway)
        SmartSignalBox(
            modifier = Modifier
                .align(Alignment.CenterStart)
                .offset(x = 50.dp, y = (-25).dp),
            direction = "W",
            state = when (uiState.currentPhase) {
                TrafficPhase.EW_GREEN -> TrafficLightState.GREEN
                TrafficPhase.EW_YELLOW -> TrafficLightState.YELLOW
                else -> TrafficLightState.RED
            },
            timeRemaining = if (uiState.currentPhase == TrafficPhase.EW_GREEN || uiState.currentPhase == TrafficPhase.EW_YELLOW) uiState.timeRemaining else 0
        )
    }
}

@Composable
private fun SmartSignalBox(
    modifier: Modifier = Modifier,
    direction: String,
    state: TrafficLightState,
    timeRemaining: Int
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = Color.Black.copy(alpha = 0.9f)
        ),
        shape = RoundedCornerShape(12.dp),
        border = BorderStroke(
            1.dp,
            Brush.linearGradient(
                colors = listOf(
                    state.color.copy(alpha = 0.6f),
                    state.color.copy(alpha = 0.2f)
                )
            )
        )
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                direction,
                color = Color.White,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold
            )

            Spacer(modifier = Modifier.height(6.dp))

            // Modern traffic light with glow effect
            Box(
                modifier = Modifier.size(20.dp),
                contentAlignment = Alignment.Center
            ) {
                // Outer glow
                Box(
                    modifier = Modifier
                        .size(20.dp)
                        .background(
                            Brush.radialGradient(
                                colors = listOf(
                                    state.color.copy(alpha = 0.4f),
                                    Color.Transparent
                                ),
                                radius = 30f
                            ),
                            CircleShape
                        )
                )

                // Inner light
                Box(
                    modifier = Modifier
                        .size(14.dp)
                        .background(
                            Brush.radialGradient(
                                colors = listOf(
                                    state.color,
                                    state.color.copy(alpha = 0.7f)
                                )
                            ),
                            CircleShape
                        )
                )
            }

            if (timeRemaining > 0) {
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    "$timeRemaining",
                    color = state.color,
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}

@Composable
fun DynamicVehicleFlow(viewModelMonitoring: StateFlow<MonitoringUiState>, vehicleCount: String) {
    val uiState by viewModelMonitoring.collectAsState()

    var vehicles by remember { mutableStateOf<List<Vehicle>>(emptyList()) }
    var currentPhase by remember { mutableStateOf(TrafficPhase.NS_GREEN) }

    println(vehicleCount.toString())

    // Update traffic phase based on signal state
    LaunchedEffect(uiState.liveSignal) {
        uiState.liveSignal?.let { signal ->
            currentPhase = determineTrafficPhase(signal)
        }
    }

    LaunchedEffect(uiState.liveSignal, vehicleCount, currentPhase) {
        uiState.liveSignal?.let { signal ->
            vehicles = generateVehiclesFromSignal(signal, vehicleCount, currentPhase)
        }
    }

    // Animate vehicles on green lanes only
    LaunchedEffect(currentPhase) {
        while (true) {
            vehicles = vehicles.map { vehicle ->
                if (shouldVehicleMove(vehicle, currentPhase)) {
                    updateVehiclePosition(vehicle)
                } else {
                    // Keep vehicles stationary on red signal
                    vehicle
                }
            }.filter {
                // Remove vehicles that have moved off screen on green lanes
                if (shouldVehicleMove(it, currentPhase)) {
                    it.position < 1.2f
                } else {
                    true // Keep all vehicles on red lanes
                }
            }
            delay(100)
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        vehicles.forEach { vehicle ->
            val (x, y) = getVehicleScreenPosition(vehicle)
            ModernVehicle(
                modifier = Modifier.offset(x.dp, y.dp),
                vehicle = vehicle,
                isMoving = shouldVehicleMove(vehicle, currentPhase)
            )
        }

        // Traffic phase indicator
        TrafficPhaseIndicator(
            modifier = Modifier.align(Alignment.Center),
            currentPhase = currentPhase
        )
    }
}

fun determineTrafficPhase(signal: ModelLiveSignal): TrafficPhase {
    return when {
        (signal.north.type == SignalType.GREEN || signal.south.type == SignalType.GREEN) &&
                (signal.east.type == SignalType.RED || signal.west.type == SignalType.RED) -> TrafficPhase.NS_GREEN

        (signal.north.type == SignalType.YELLOW || signal.south.type == SignalType.YELLOW) &&
                (signal.east.type == SignalType.RED || signal.west.type == SignalType.RED) -> TrafficPhase.NS_YELLOW

        (signal.east.type == SignalType.GREEN || signal.west.type == SignalType.GREEN) &&
                (signal.north.type == SignalType.RED || signal.south.type == SignalType.RED) -> TrafficPhase.EW_GREEN

        (signal.east.type == SignalType.YELLOW || signal.west.type == SignalType.YELLOW) &&
                (signal.north.type == SignalType.RED || signal.south.type == SignalType.RED) -> TrafficPhase.EW_YELLOW

        else -> TrafficPhase.NS_GREEN // Default
    }
}

fun shouldVehicleMove(vehicle: Vehicle, currentPhase: TrafficPhase): Boolean {
    return when (currentPhase) {
        TrafficPhase.NS_GREEN, TrafficPhase.NS_YELLOW -> {
            // North-South lanes can move
            vehicle.lane in listOf(
                VehicleLane.NORTH_1, VehicleLane.NORTH_2, VehicleLane.NORTH_3,
                VehicleLane.SOUTH_1, VehicleLane.SOUTH_2, VehicleLane.SOUTH_3
            )
        }
        TrafficPhase.EW_GREEN, TrafficPhase.EW_YELLOW -> {
            // East-West lanes can move
            vehicle.lane in listOf(
                VehicleLane.EAST_1, VehicleLane.EAST_2, VehicleLane.EAST_3,
                VehicleLane.WEST_1, VehicleLane.WEST_2, VehicleLane.WEST_3
            )
        }

        TrafficPhase.ALL_RED -> {
            vehicle.lane in listOf(
                VehicleLane.EAST_1, VehicleLane.EAST_2, VehicleLane.EAST_3,
                VehicleLane.WEST_1, VehicleLane.WEST_2, VehicleLane.WEST_3
            )
        }
    }
}

fun generateVehiclesFromSignal(signal: ModelLiveSignal, vehicleCountJson: String, currentPhase: TrafficPhase): List<Vehicle> {
    val parsed = Json.decodeFromString<VehicleCountJson>(vehicleCountJson)
    val vehicleTypeMap = mutableMapOf<VehicleType, Int>().apply {
        VehicleType.entries.forEach { type ->
            val count = when (type.name.lowercase()) {
                "car" -> parsed.car
                "motorcycle" -> parsed.motorcycle
                "bus" -> parsed.bus
                "truck" -> parsed.truck
                "person" -> parsed.person
                else -> 0
            }
            this[type] = count
        }
    }

    // Separate lanes based on current traffic phase
    val (greenLanes, redLanes) = when (currentPhase) {
        TrafficPhase.NS_GREEN, TrafficPhase.NS_YELLOW -> {
            val green = listOf(
                VehicleLane.NORTH_1, VehicleLane.NORTH_2, VehicleLane.NORTH_3,
                VehicleLane.SOUTH_1, VehicleLane.SOUTH_2, VehicleLane.SOUTH_3
            )
            val red = listOf(
                VehicleLane.EAST_1, VehicleLane.EAST_2, VehicleLane.EAST_3,
                VehicleLane.WEST_1, VehicleLane.WEST_2, VehicleLane.WEST_3
            )
            Pair(green, red)
        }
        TrafficPhase.EW_GREEN, TrafficPhase.EW_YELLOW -> {
            val green = listOf(
                VehicleLane.EAST_1, VehicleLane.EAST_2, VehicleLane.EAST_3,
                VehicleLane.WEST_1, VehicleLane.WEST_2, VehicleLane.WEST_3
            )
            val red = listOf(
                VehicleLane.NORTH_1, VehicleLane.NORTH_2, VehicleLane.NORTH_3,
                VehicleLane.SOUTH_1, VehicleLane.SOUTH_2, VehicleLane.SOUTH_3
            )
            Pair(green, red)
        }

        TrafficPhase.ALL_RED -> {
            val green = listOf(
                VehicleLane.EAST_1, VehicleLane.EAST_2, VehicleLane.EAST_3,
                VehicleLane.WEST_1, VehicleLane.WEST_2, VehicleLane.WEST_3
            )
            val red = listOf(
                VehicleLane.NORTH_1, VehicleLane.NORTH_2, VehicleLane.NORTH_3,
                VehicleLane.SOUTH_1, VehicleLane.SOUTH_2, VehicleLane.SOUTH_3
            )
            Pair(green, red)
        }
    }

    var idCounter = 0
    val vehicles = mutableListOf<Vehicle>()

    // Calculate vehicles per lane type
    val totalVehicleCount = vehicleTypeMap.values.sum()
    val vehiclesPerDirection = totalVehicleCount / 2 // Split between two directions

    // Distribute vehicles to red lanes (waiting vehicles)
    var redLaneIndex = 0
    for ((type, count) in vehicleTypeMap) {
        val vehiclesForRedLanes = (count * 0.6).toInt() // 60% waiting on red
        repeat(vehiclesForRedLanes) {
            val lane = redLanes[redLaneIndex % redLanes.size]
            redLaneIndex++

            // Position vehicles in a queue (closer to intersection)
            val queuePosition = (redLaneIndex % 8) * 0.08f + 0.6f

            vehicles.add(
                Vehicle(
                    id = idCounter++,
                    lane = lane,
                    type = type,
                    position = queuePosition,
                    speed = Random.nextFloat() * 0.01f + 0.005f,
                    color = type.color
                )
            )
        }
    }

    // Distribute vehicles to green lanes (moving vehicles)
    var greenLaneIndex = 0
    for ((type, count) in vehicleTypeMap) {
        val vehiclesForGreenLanes = (count * 0.4).toInt() // 40% moving on green
        repeat(vehiclesForGreenLanes) {
            val lane = greenLanes[greenLaneIndex % greenLanes.size]
            greenLaneIndex++

            // Spread vehicles across the green lanes
            val movingPosition = Random.nextFloat() * 0.8f

            vehicles.add(
                Vehicle(
                    id = idCounter++,
                    lane = lane,
                    type = type,
                    position = movingPosition,
                    speed = Random.nextFloat() * 0.01f + 0.005f,
                    color = type.color
                )
            )
        }
    }

    return vehicles
}

data class Vehicle(
    val id: Int,
    val lane: VehicleLane,
    val type: VehicleType,
    val position: Float,
    val speed: Float,
    val color: Color
)

enum class VehicleLane {
    EAST_1, EAST_2, EAST_3,  // Left to right movement
    WEST_1, WEST_2, WEST_3,  // Right to left movement
    SOUTH_1, SOUTH_2, SOUTH_3,  // Top to bottom movement
    NORTH_1, NORTH_2, NORTH_3   // Bottom to top movement
}

enum class VehicleType(val color: Color, val icon: ImageVector) {
    CAR(Color(0xFF3B82F6), Icons.Default.DirectionsCar),
    TAXI(Color(0xFFFBBF24), Icons.Default.LocalTaxi),
    BUS(Color(0xFF059669), Icons.Default.DirectionsBus),
    TRUCK(Color(0xFFDC2626), Icons.Default.LocalShipping),
    BIKE(Color(0xFF8B5CF6), Icons.Default.DirectionsBike)
}

private fun updateVehiclePosition(vehicle: Vehicle): Vehicle {
    return vehicle.copy(position = vehicle.position + vehicle.speed)
}

private fun getVehicleScreenPosition(vehicle: Vehicle): Pair<Float, Float> {
    val progress = vehicle.position
    val roadCenter = 140f // Center of the intersection area
    val laneOffset = 27f // Distance between lanes

    return when (vehicle.lane) {
        // Broadway Eastbound (Left to Right)
        VehicleLane.EAST_1 -> Pair(-30f + progress * 400f, roadCenter - laneOffset)
        VehicleLane.EAST_2 -> Pair(-30f + progress * 400f, roadCenter)
        VehicleLane.EAST_3 -> Pair(-30f + progress * 400f, roadCenter + laneOffset)

        // Broadway Westbound (Right to Left)
        VehicleLane.WEST_1 -> Pair(370f - progress * 400f, roadCenter + laneOffset + 15f)
        VehicleLane.WEST_2 -> Pair(370f - progress * 400f, roadCenter + 15f)
        VehicleLane.WEST_3 -> Pair(370f - progress * 400f, roadCenter - laneOffset + 15f)

        // 42nd Street Southbound (Top to Bottom)
        VehicleLane.SOUTH_1 -> Pair(roadCenter - laneOffset, -30f + progress * 340f)
        VehicleLane.SOUTH_2 -> Pair(roadCenter, -30f + progress * 340f)
        VehicleLane.SOUTH_3 -> Pair(roadCenter + laneOffset, -30f + progress * 340f)

        // 42nd Street Northbound (Bottom to Top)
        VehicleLane.NORTH_1 -> Pair(roadCenter + laneOffset + 15f, 310f - progress * 340f)
        VehicleLane.NORTH_2 -> Pair(roadCenter + 15f, 310f - progress * 340f)
        VehicleLane.NORTH_3 -> Pair(roadCenter - laneOffset + 15f, 310f - progress * 340f)
    }
}

@Composable
private fun ModernVehicle(
    modifier: Modifier = Modifier,
    vehicle: Vehicle,
    isMoving: Boolean = true
) {
    Box(modifier = modifier) {
        // Vehicle shadow/glow - different for moving vs stopped
        Box(
            modifier = Modifier
                .size(14.dp)
                .background(
                    Brush.radialGradient(
                        colors = if (isMoving) {
                            listOf(
                                vehicle.color.copy(alpha = 0.5f),
                                Color.Transparent
                            )
                        } else {
                            listOf(
                                Color.Red.copy(alpha = 0.3f),
                                Color.Transparent
                            )
                        },
                        radius = 25f
                    ),
                    RoundedCornerShape(3.dp)
                )
        )

        // Vehicle body
        Box(
            modifier = Modifier
                .align(Alignment.Center)
                .size(10.dp, 6.dp)
                .background(
                    if (isMoving) vehicle.color else vehicle.color.copy(alpha = 0.7f),
                    RoundedCornerShape(2.dp)
                ),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                vehicle.type.icon,
                contentDescription = null,
                tint = Color.White,
                modifier = Modifier.size(5.dp)
            )
        }

        // Status indicator - green for moving, red for stopped
        Box(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .size(3.dp)
                .background(
                    if (isMoving) Color(0xFF00FF88) else Color(0xFFFF4444),
                    CircleShape
                )
        )
    }
}

@Composable
private fun TrafficPhaseIndicator(
    modifier: Modifier = Modifier,
    currentPhase: TrafficPhase
) {
    Card(
        modifier = modifier.size(60.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        shape = RoundedCornerShape(12.dp),
        border = BorderStroke(1.dp, getCurrentPhaseColor(currentPhase).copy(alpha = 0.4f))
    ) {
        Column(
            modifier = Modifier.fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Icon(
                Icons.Default.Traffic,
                contentDescription = null,
                tint = getCurrentPhaseColor(currentPhase),
                modifier = Modifier.size(16.dp)
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                getCurrentPhaseText(currentPhase),
                color = getCurrentPhaseColor(currentPhase),
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold
            )
            Text(
                getPhaseStatus(currentPhase),
                color = Color.White,
                fontSize = 8.sp
            )
        }
    }
}

private fun getCurrentPhaseColor(phase: TrafficPhase): Color {
    return when (phase) {
        TrafficPhase.NS_GREEN -> Color(0xFF10B981)
        TrafficPhase.NS_YELLOW -> Color(0xFFFBBF24)
        TrafficPhase.EW_GREEN -> Color(0xFF10B981)
        TrafficPhase.EW_YELLOW -> Color(0xFFFBBF24)
        TrafficPhase.ALL_RED -> Color(0xFFE20000)
    }
}

private fun getCurrentPhaseText(phase: TrafficPhase): String {
    return when (phase) {
        TrafficPhase.NS_GREEN, TrafficPhase.NS_YELLOW -> "N-S"
        TrafficPhase.EW_GREEN, TrafficPhase.EW_YELLOW -> "E-W"
        TrafficPhase.ALL_RED -> ""
    }
}

private fun getPhaseStatus(phase: TrafficPhase): String {
    return when (phase) {
        TrafficPhase.NS_GREEN, TrafficPhase.EW_GREEN -> "GREEN"
        TrafficPhase.NS_YELLOW, TrafficPhase.EW_YELLOW -> "YELLOW"
        TrafficPhase.ALL_RED -> "RED"
    }
}

enum class TrafficPhase {
    NS_GREEN, NS_YELLOW, EW_GREEN, EW_YELLOW, ALL_RED
}

@Composable
private fun StatsView() {
    LazyColumn(
        modifier = Modifier
            .fillMaxWidth()
            .height(280.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item {
            StatCard(
                title = "Peak Hour Analysis",
                value = "8:00-9:00 AM",
                detail = "Highest traffic volume: 2,847 vehicles/hour",
                color = Color(0xFFEF4444)
            )
        }
        item {
            StatCard(
                title = "Average Wait Time",
                value = "3.2 minutes",
                detail = "Down 15% from last week",
                color = Color(0xFF10B981)
            )
        }
        item {
            StatCard(
                title = "Traffic Efficiency",
                value = "87%",
                detail = "Optimal signal timing achieved",
                color = Color(0xFF3B82F6)
            )
        }
    }
}

@Composable
private fun HistoryView() {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .height(280.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            "Recent Events",
            color = Color.White,
            fontSize = 14.sp,
            fontWeight = FontWeight.Bold
        )

        LazyColumn(
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            items(5) { index ->
                HistoryItem(
                    time = "09:${18-index}:${36-index*6}",
                    event = when (index) {
                        0 -> "Signal optimization applied"
                        1 -> "Heavy traffic detected on Broadway"
                        2 -> "Emergency vehicle priority activated"
                        3 -> "Pedestrian crossing extended"
                        else -> "Normal traffic flow resumed"
                    },
                    type = when (index) {
                        0 -> "SYSTEM"
                        1 -> "ALERT"
                        2 -> "PRIORITY"
                        3 -> "PEDESTRIAN"
                        else -> "INFO"
                    }
                )
            }
        }
    }
}

@Composable
private fun StatCard(
    title: String,
    value: String,
    detail: String,
    color: Color
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = color.copy(alpha = 0.1f)
        ),
        shape = RoundedCornerShape(12.dp),
        border = BorderStroke(1.dp, color.copy(alpha = 0.2f))
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                title,
                color = color,
                fontSize = 11.sp,
                fontWeight = FontWeight.Medium
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                value,
                color = Color.White,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold
            )
            Text(
                detail,
                color = Color(0xFF9CA3AF),
                fontSize = 10.sp
            )
        }
    }
}

@Composable
private fun HistoryItem(
    time: String,
    event: String,
    type: String
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(
                Color(0xFF1F2937).copy(alpha = 0.5f),
                RoundedCornerShape(8.dp)
            )
            .padding(12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Badge(
            containerColor = when (type) {
                "ALERT" -> Color(0xFFEF4444).copy(alpha = 0.2f)
                "PRIORITY" -> Color(0xFFF59E0B).copy(alpha = 0.2f)
                "SYSTEM" -> Color(0xFF3B82F6).copy(alpha = 0.2f)
                else -> Color(0xFF6B7280).copy(alpha = 0.2f)
            },
            contentColor = when (type) {
                "ALERT" -> Color(0xFFEF4444)
                "PRIORITY" -> Color(0xFFF59E0B)
                "SYSTEM" -> Color(0xFF3B82F6)
                else -> Color(0xFF9CA3AF)
            }
        ) {
            Text(
                type,
                fontSize = 8.sp,
                fontWeight = FontWeight.Bold
            )
        }

        Spacer(modifier = Modifier.width(8.dp))

        Column(modifier = Modifier.weight(1f)) {
            Text(
                event,
                color = Color.White,
                fontSize = 11.sp,
                fontWeight = FontWeight.Medium
            )
            Text(
                time,
                color = Color(0xFF6B7280),
                fontSize = 9.sp
            )
        }
    }
}

@Composable
private fun QuickMetrics(vehicleCount: String) {
    val vehicle = Json.decodeFromString<VehicleCountJson>(vehicleCount)
    val count = vehicle.car + vehicle.motorcycle + vehicle.bus + vehicle.truck // + vehicle.person
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        QuickMetric(
            modifier = Modifier.weight(1f),
            label = "VEHICLES",
            value = count.toString(),
            trend = "+12%",
            color = Color(0xFF00D4FF)
        )

        QuickMetric(
            modifier = Modifier.weight(1f),
            label = "WAIT TIME",
            value = "~ m", //"3.2m",
            trend = "~%", //"-8%",
            color = Color(0xFF10B981)
        )

        QuickMetric(
            modifier = Modifier.weight(1f),
            label = "EFFICIENCY",
            value = "~%", //"87%",
            trend = "~%", //"+5%",
            color = Color(0xFFFBBF24)
        )
    }
}

@Composable
private fun QuickMetric(
    modifier: Modifier = Modifier,
    label: String,
    value: String,
    trend: String,
    color: Color
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        ),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                label,
                color = Color(0xFF9CA3AF),
                fontSize = 9.sp,
                fontWeight = FontWeight.Medium,
                letterSpacing = 0.5.sp
            )

            Spacer(modifier = Modifier.height(2.dp))

            Text(
                value,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold
            )

            Text(
                trend,
                color = color,
                fontSize = 10.sp,
                fontWeight = FontWeight.Medium
            )
        }
    }
}

//@Composable
//fun DynamicVehicleFlow(viewModelMonitoring: StateFlow<MonitoringUiState>, vehicleCount: String) {
//    val uiState by viewModelMonitoring.collectAsState()
//
//    var vehicles by remember { mutableStateOf<List<Vehicle>>(emptyList()) }
//
//    println(vehicleCount.toString())
//
//    LaunchedEffect(uiState.liveSignal, vehicleCount) {
//        uiState.liveSignal?.let { signal ->
//            vehicles = generateVehiclesFromSignal(signal, vehicleCount)
//        }
//    }
//
////    LaunchedEffect(Unit) {
////        while (true) {
////            vehicles = vehicles.map { vehicle ->
////                updateVehiclePosition(vehicle)
////            }.filter { it.position < 1.2f }
////
////            delay(100)
////        }
////    }
//
//    Box(modifier = Modifier.fillMaxSize()) {
//        vehicles.forEach { vehicle ->
//            val (x, y) = getVehicleScreenPosition(vehicle)
//            ModernVehicle(
//                modifier = Modifier.offset(x.dp, y.dp),
//                vehicle = vehicle
//            )
//        }
//    }
//}
//
//fun generateVehiclesFromSignal(signal: ModelLiveSignal, vehicleCountJson: String): List<Vehicle> {
//    val parsed = Json.decodeFromString<VehicleCountJson>(vehicleCountJson)
//    val vehicleTypeMap = mutableMapOf<VehicleType, Int>().apply {
//        VehicleType.entries.forEach { type ->
//            val count = when (type.name.lowercase()) {
//                "car" -> parsed.car
//                "motorcycle" -> parsed.motorcycle
//                "bus" -> parsed.bus
//                "truck" -> parsed.truck
//                "person" -> parsed.person
//                else -> 0
//            }
//            this[type] = count
//        }
//    }
//
//    val roadToLanes = mapOf(
//        VehicleLane.EAST_1 to signal.east,
//        VehicleLane.EAST_2 to signal.east,
//        VehicleLane.EAST_3 to signal.east,
//
//        VehicleLane.WEST_1 to signal.west,
//        VehicleLane.WEST_2 to signal.west,
//        VehicleLane.WEST_3 to signal.west,
//
//        VehicleLane.SOUTH_1 to signal.south,
//        VehicleLane.SOUTH_2 to signal.south,
//        VehicleLane.SOUTH_3 to signal.south,
//
//        VehicleLane.NORTH_1 to signal.north,
//        VehicleLane.NORTH_2 to signal.north,
//        VehicleLane.NORTH_3 to signal.north,
//    )
//
//    val allLanes = roadToLanes.keys.toList()
//    val totalVehicleCount = vehicleTypeMap.values.sum()
//    var idCounter = 0
//    var laneIndex = 0
//
//    val vehicles = mutableListOf<Vehicle>()
//
//    for ((type, count) in vehicleTypeMap) {
//        repeat(count) {
//            val lane = allLanes[laneIndex % allLanes.size]
//            laneIndex++
//            vehicles.add(
//                Vehicle(
//                    id = idCounter++,
//                    lane = lane,
//                    type = type,
//                    position = 0f,
//                    speed = Random.nextFloat() * 0.01f + 0.005f,
//                    color = type.color
//                )
//            )
//        }
//    }
//
//    return vehicles
//}
//
//data class Vehicle(
//    val id: Int,
//    val lane: VehicleLane,
//    val type: VehicleType,
//    val position: Float,
//    val speed: Float,
//    val color: Color
//)
//
//
//enum class VehicleLane {
//    EAST_1, EAST_2, EAST_3,  // Left to right movement
//    WEST_1, WEST_2, WEST_3,  // Right to left movement
//    SOUTH_1, SOUTH_2, SOUTH_3,  // Top to bottom movement
//    NORTH_1, NORTH_2, NORTH_3   // Bottom to top movement
//}
//
//enum class VehicleType(val color: Color, val icon: ImageVector) {
//    CAR(Color(0xFF3B82F6), Icons.Default.DirectionsCar),
//    TAXI(Color(0xFFFBBF24), Icons.Default.LocalTaxi),
//    BUS(Color(0xFF059669), Icons.Default.DirectionsBus),
//    TRUCK(Color(0xFFDC2626), Icons.Default.LocalShipping),
//    BIKE(Color(0xFF8B5CF6), Icons.Default.DirectionsBike)
//}
//
//private fun generateRandomVehicles(): List<Vehicle> {
//    return (1..10).map { generateRandomVehicle() }
//}
//
//private fun generateRandomVehicle(): Vehicle {
//    val lanes = VehicleLane.values()
//    val types = VehicleType.values()
//    val selectedType = types.random()
//
//    return Vehicle(
//        id = Random.nextInt(),
//        lane = lanes.random(),
//        type = selectedType,
//        position = Random.nextFloat() * 0.2f,
//        speed = Random.nextFloat() * 0.015f + 0.008f,
//        color = selectedType.color
//    )
//}
//
//private fun updateVehiclePosition(vehicle: Vehicle): Vehicle {
//    return vehicle.copy(position = vehicle.position + vehicle.speed)
//}
//
//private fun getVehicleScreenPosition(vehicle: Vehicle): Pair<Float, Float> {
//    val progress = vehicle.position
//    val roadCenter = 140f // Center of the intersection area
//    val laneOffset = 27f // Distance between lanes
//
//    return when (vehicle.lane) {
//        // Broadway Eastbound (Left to Right)
//        VehicleLane.EAST_1 -> Pair(-30f + progress * 400f, roadCenter - laneOffset)
//        VehicleLane.EAST_2 -> Pair(-30f + progress * 400f, roadCenter)
//        VehicleLane.EAST_3 -> Pair(-30f + progress * 400f, roadCenter + laneOffset)
//
//        // Broadway Westbound (Right to Left)
//        VehicleLane.WEST_1 -> Pair(370f - progress * 400f, roadCenter + laneOffset + 15f)
//        VehicleLane.WEST_2 -> Pair(370f - progress * 400f, roadCenter + 15f)
//        VehicleLane.WEST_3 -> Pair(370f - progress * 400f, roadCenter - laneOffset + 15f)
//
//        // 42nd Street Southbound (Top to Bottom)
//        VehicleLane.SOUTH_1 -> Pair(roadCenter - laneOffset, -30f + progress * 340f)
//        VehicleLane.SOUTH_2 -> Pair(roadCenter, -30f + progress * 340f)
//        VehicleLane.SOUTH_3 -> Pair(roadCenter + laneOffset, -30f + progress * 340f)
//
//        // 42nd Street Northbound (Bottom to Top)
//        VehicleLane.NORTH_1 -> Pair(roadCenter + laneOffset + 15f, 310f - progress * 340f)
//        VehicleLane.NORTH_2 -> Pair(roadCenter + 15f, 310f - progress * 340f)
//        VehicleLane.NORTH_3 -> Pair(roadCenter - laneOffset + 15f, 310f - progress * 340f)
//    }
//}
//
//@Composable
//private fun ModernVehicle(
//    modifier: Modifier = Modifier,
//    vehicle: Vehicle
//) {
//    Box(modifier = modifier) {
//        // Vehicle shadow/glow
//        Box(
//            modifier = Modifier
//                .size(14.dp)
//                .background(
//                    Brush.radialGradient(
//                        colors = listOf(
//                            vehicle.color.copy(alpha = 0.5f),
//                            Color.Transparent
//                        ),
//                        radius = 25f
//                    ),
//                    RoundedCornerShape(3.dp)
//                )
//        )
//
//        // Vehicle body
//        Box(
//            modifier = Modifier
//                .align(Alignment.Center)
//                .size(10.dp, 6.dp)
//                .background(
//                    vehicle.color,
//                    RoundedCornerShape(2.dp)
//                ),
//            contentAlignment = Alignment.Center
//        ) {
//            Icon(
//                vehicle.type.icon,
//                contentDescription = null,
//                tint = Color.White,
//                modifier = Modifier.size(5.dp)
//            )
//        }
//
//        // Direction indicator based on lane
//        val directionColor = when (vehicle.lane) {
//            VehicleLane.EAST_1, VehicleLane.EAST_2, VehicleLane.EAST_3 -> Color(0xFF00FF88)
//            VehicleLane.WEST_1, VehicleLane.WEST_2, VehicleLane.WEST_3 -> Color(0xFFFF6B6B)
//            VehicleLane.SOUTH_1, VehicleLane.SOUTH_2, VehicleLane.SOUTH_3 -> Color(0xFF4ECDC4)
//            VehicleLane.NORTH_1, VehicleLane.NORTH_2, VehicleLane.NORTH_3 -> Color(0xFFFFE66D)
//        }
//
//        Box(
//            modifier = Modifier
//                .align(Alignment.TopEnd)
//                .size(3.dp)
//                .background(directionColor, CircleShape)
//        )
//    }
//}
//
//@Composable
//private fun BoxScope.TrafficInfoOverlay() {
//    // Current phase indicator in center
//    Card(
//        modifier = Modifier
//            .align(Alignment.Center)
//            .size(60.dp),
//        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
//        shape = RoundedCornerShape(12.dp),
//        border = BorderStroke(1.dp, Color(0xFF3B82F6).copy(alpha = 0.4f))
//    ) {
//        Column(
//            modifier = Modifier.fillMaxSize(),
//            horizontalAlignment = Alignment.CenterHorizontally,
//            verticalArrangement = Arrangement.Center
//        ) {
//            Icon(
//                Icons.Default.Traffic,
//                contentDescription = null,
//                tint = Color(0xFF10B981),
//                modifier = Modifier.size(16.dp)
//            )
//            Spacer(modifier = Modifier.height(2.dp))
//            Text(
//                "N-S",
//                color = Color(0xFF10B981),
//                fontSize = 10.sp,
//                fontWeight = FontWeight.Bold
//            )
//            Text(
//                "ACTIVE",
//                color = Color.White,
//                fontSize = 8.sp
//            )
//        }
//    }
//}
//
//@Composable
//private fun StatsView() {
//    LazyColumn(
//        modifier = Modifier
//            .fillMaxWidth()
//            .height(280.dp),
//        verticalArrangement = Arrangement.spacedBy(12.dp)
//    ) {
//        item {
//            StatCard(
//                title = "Peak Hour Analysis",
//                value = "8:00-9:00 AM",
//                detail = "Highest traffic volume: 2,847 vehicles/hour",
//                color = Color(0xFFEF4444)
//            )
//        }
//        item {
//            StatCard(
//                title = "Average Wait Time",
//                value = "3.2 minutes",
//                detail = "Down 15% from last week",
//                color = Color(0xFF10B981)
//            )
//        }
//        item {
//            StatCard(
//                title = "Traffic Efficiency",
//                value = "87%",
//                detail = "Optimal signal timing achieved",
//                color = Color(0xFF3B82F6)
//            )
//        }
//    }
//}
//
//@Composable
//private fun HistoryView() {
//    Column(
//        modifier = Modifier
//            .fillMaxWidth()
//            .height(280.dp),
//        verticalArrangement = Arrangement.spacedBy(8.dp)
//    ) {
//        Text(
//            "Recent Events",
//            color = Color.White,
//            fontSize = 14.sp,
//            fontWeight = FontWeight.Bold
//        )
//
//        LazyColumn(
//            verticalArrangement = Arrangement.spacedBy(6.dp)
//        ) {
//            items(5) { index ->
//                HistoryItem(
//                    time = "09:${18-index}:${36-index*6}",
//                    event = when (index) {
//                        0 -> "Signal optimization applied"
//                        1 -> "Heavy traffic detected on Broadway"
//                        2 -> "Emergency vehicle priority activated"
//                        3 -> "Pedestrian crossing extended"
//                        else -> "Normal traffic flow resumed"
//                    },
//                    type = when (index) {
//                        0 -> "SYSTEM"
//                        1 -> "ALERT"
//                        2 -> "PRIORITY"
//                        3 -> "PEDESTRIAN"
//                        else -> "INFO"
//                    }
//                )
//            }
//        }
//    }
//}
//
//@Composable
//private fun StatCard(
//    title: String,
//    value: String,
//    detail: String,
//    color: Color
//) {
//    Card(
//        colors = CardDefaults.cardColors(
//            containerColor = color.copy(alpha = 0.1f)
//        ),
//        shape = RoundedCornerShape(12.dp),
//        border = BorderStroke(1.dp, color.copy(alpha = 0.2f))
//    ) {
//        Column(
//            modifier = Modifier.padding(16.dp)
//        ) {
//            Text(
//                title,
//                color = color,
//                fontSize = 11.sp,
//                fontWeight = FontWeight.Medium
//            )
//            Spacer(modifier = Modifier.height(4.dp))
//            Text(
//                value,
//                color = Color.White,
//                fontSize = 16.sp,
//                fontWeight = FontWeight.Bold
//            )
//            Text(
//                detail,
//                color = Color(0xFF9CA3AF),
//                fontSize = 10.sp
//            )
//        }
//    }
//}
//
//@Composable
//private fun HistoryItem(
//    time: String,
//    event: String,
//    type: String
//) {
//    Row(
//        modifier = Modifier
//            .fillMaxWidth()
//            .background(
//                Color(0xFF1F2937).copy(alpha = 0.5f),
//                RoundedCornerShape(8.dp)
//            )
//            .padding(12.dp),
//        verticalAlignment = Alignment.CenterVertically
//    ) {
//        Badge(
//            containerColor = when (type) {
//                "ALERT" -> Color(0xFFEF4444).copy(alpha = 0.2f)
//                "PRIORITY" -> Color(0xFFF59E0B).copy(alpha = 0.2f)
//                "SYSTEM" -> Color(0xFF3B82F6).copy(alpha = 0.2f)
//                else -> Color(0xFF6B7280).copy(alpha = 0.2f)
//            },
//            contentColor = when (type) {
//                "ALERT" -> Color(0xFFEF4444)
//                "PRIORITY" -> Color(0xFFF59E0B)
//                "SYSTEM" -> Color(0xFF3B82F6)
//                else -> Color(0xFF9CA3AF)
//            }
//        ) {
//            Text(
//                type,
//                fontSize = 8.sp,
//                fontWeight = FontWeight.Bold
//            )
//        }
//
//        Spacer(modifier = Modifier.width(8.dp))
//
//        Column(modifier = Modifier.weight(1f)) {
//            Text(
//                event,
//                color = Color.White,
//                fontSize = 11.sp,
//                fontWeight = FontWeight.Medium
//            )
//            Text(
//                time,
//                color = Color(0xFF6B7280),
//                fontSize = 9.sp
//            )
//        }
//    }
//}
//
//@Composable
//private fun QuickMetrics() {
//    Row(
//        modifier = Modifier.fillMaxWidth(),
//        horizontalArrangement = Arrangement.spacedBy(12.dp)
//    ) {
//        QuickMetric(
//            modifier = Modifier.weight(1f),
//            label = "VEHICLES",
//            value = "1,247",
//            trend = "+12%",
//            color = Color(0xFF00D4FF)
//        )
//
//        QuickMetric(
//            modifier = Modifier.weight(1f),
//            label = "WAIT TIME",
//            value = "3.2m",
//            trend = "-8%",
//            color = Color(0xFF10B981)
//        )
//
//        QuickMetric(
//            modifier = Modifier.weight(1f),
//            label = "EFFICIENCY",
//            value = "87%",
//            trend = "+5%",
//            color = Color(0xFFFBBF24)
//        )
//    }
//}
//
//@Composable
//private fun QuickMetric(
//    modifier: Modifier = Modifier,
//    label: String,
//    value: String,
//    trend: String,
//    color: Color
//) {
//    Card(
//        modifier = modifier,
//        colors = CardDefaults.cardColors(
//            containerColor = MaterialTheme.colorScheme.surfaceVariant
//        ),
//        shape = RoundedCornerShape(12.dp)
//    ) {
//        Column(
//            modifier = Modifier.padding(12.dp),
//            horizontalAlignment = Alignment.CenterHorizontally
//        ) {
//            Text(
//                label,
//                color = Color(0xFF9CA3AF),
//                fontSize = 9.sp,
//                fontWeight = FontWeight.Medium,
//                letterSpacing = 0.5.sp
//            )
//
//            Spacer(modifier = Modifier.height(2.dp))
//
//            Text(
//                value,
//                color = MaterialTheme.colorScheme.onSurfaceVariant,
//                fontSize = 16.sp,
//                fontWeight = FontWeight.Bold
//            )
//
//            Text(
//                trend,
//                color = color,
//                fontSize = 10.sp,
//                fontWeight = FontWeight.Medium
//            )
//        }
//    }
//}
//
//enum class TrafficPhase {
//    NS_GREEN, NS_YELLOW, EW_GREEN, EW_YELLOW
//}

enum class TrafficLightState(val color: Color) {
    RED(Color(0xFFEF4444)),
    YELLOW(Color(0xFFF59E0B)),
    GREEN(Color(0xFF10B981))
}