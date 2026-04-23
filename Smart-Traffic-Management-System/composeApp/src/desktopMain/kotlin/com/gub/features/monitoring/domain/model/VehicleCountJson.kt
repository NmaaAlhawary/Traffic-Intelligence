package com.gub.features.monitoring.domain.model

import kotlinx.serialization.*
import kotlinx.serialization.json.*

@Serializable
data class VehicleCountJson(
    val person: Int = 0,
    val car: Int = 0,
    val motorcycle: Int = 0,
    val bus: Int = 0,
    val truck: Int = 0
)