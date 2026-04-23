package com.gub.data.service.monitoring

import com.gub.domain.models.monitoring.ModelLiveSignal.Road

class ServiceLiveSignal {

    fun eastState() : Road {
        return Road(
            name = "East Road",
            type = Road.SignalType.GREEN,
            vehicleCount = 10
        )
    }

    fun westState() : Road {
        return Road(
            name = "West Road",
            type = Road.SignalType.GREEN,
            vehicleCount = 10
        )
    }

    fun northState() : Road {
        return Road(
            name = "North Road",
            type = Road.SignalType.GREEN,
            vehicleCount = 10
        )
    }

    fun southState() : Road {
        return Road(
            name = "South Road",
            type = Road.SignalType.GREEN,
            vehicleCount = 10
        )
    }

}