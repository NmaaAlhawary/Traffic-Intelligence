package com.gub.data.service.common

import com.gub.application.Config.API_KEY
import com.gub.application.Config.OPEN_WEATHER
import com.gub.data.database.dao.WeatherDao
import com.gub.data.database.entity.WeatherData.Companion.toWeatherData
import com.gub.models.dashboard.overview.ModelWeather
import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import org.slf4j.LoggerFactory
import kotlin.random.Random

class WeatherService(
    private val weatherDao: WeatherDao,
    private val defaultCity: String = "Dhaka",
    private val defaultCountryCode: String = "BD"
) {

    companion object {
        private var lastLoad: Long = 0
        private var lastWeather: ModelWeather? = null
    }

    private val logger = LoggerFactory.getLogger(WeatherService::class.java)

    private val httpClient = HttpClient(CIO) {
        install(ContentNegotiation) {
            json(Json {
                ignoreUnknownKeys = true
                isLenient = true
            })
        }
    }

    suspend fun getCurrentWeather(
        city: String = defaultCity,
        countryCode: String = defaultCountryCode
    ): ModelWeather = withContext(Dispatchers.IO) {
        if (lastLoad >= (System.currentTimeMillis() - 60 * 60 * 1000)) {
            if (lastWeather != null)
                return@withContext lastWeather!!
        }
        try {
            val response = httpClient.get(OPEN_WEATHER) {
                parameter("q", "$city,$countryCode")
                parameter("appid", API_KEY)
                parameter("units", "metric") // Get temperature in Celsius
            }

            val weatherData: OpenWeatherResponse = response.body()

            ModelWeather(
                temp = weatherData.main.temp,
                tempUnit = ModelWeather.TempUnit.CELSIUS,
                wind = weatherData.wind?.speed ?: 0.0,
                humidity = weatherData.main.humidity.toDouble(),
                visibility = (weatherData.visibility ?: 10000) / 1000.0 // Convert meters to kilometers
            ).also {
                lastWeather = it
                lastLoad = System.currentTimeMillis()
                weatherDao.insert(it.toWeatherData())
            }
        } catch (e: Exception) {
            logger.error("Failed to fetch weather data: ${e.message}", e)
            getFallbackWeather()
        }
    }

    suspend fun getCurrentWeatherByCoordinates(
        lat: Double,
        lon: Double
    ): ModelWeather = withContext(Dispatchers.IO) {
        try {
            val response = httpClient.get(OPEN_WEATHER) {
                parameter("lat", lat)
                parameter("lon", lon)
                parameter("appid", API_KEY)
                parameter("units", "metric")
            }

            val weatherData: OpenWeatherResponse = response.body()

            ModelWeather(
                temp = weatherData.main.temp,
                tempUnit = ModelWeather.TempUnit.CELSIUS,
                humidity = weatherData.main.humidity.toDouble(),
                visibility = (weatherData.visibility ?: 10000) / 1000.0
            )
        } catch (e: Exception) {
            logger.error("Failed to fetch weather data by coordinates: ${e.message}", e)
            getFallbackWeather()
        }
    }

    private fun getFallbackWeather(): ModelWeather {
        logger.info("Using fallback weather data")
        return ModelWeather(
            temp = Random.nextDouble(15.0, 35.0).round(1),
            tempUnit = ModelWeather.TempUnit.CELSIUS,
            humidity = Random.nextDouble(30.0, 90.0).round(1),
            visibility = Random.nextDouble(0.5, 10.0).round(1)
        )
    }

    fun close() {
        httpClient.close()
    }

    private fun Double.round(decimals: Int): Double {
        var multiplier = 1.0
        repeat(decimals) { multiplier *= 10 }
        return kotlin.math.round(this * multiplier) / multiplier
    }
}

@Serializable
private data class OpenWeatherResponse(
    val main: Main,
    val visibility: Int? = null,
    val wind: Wind? = null,
    val name: String
) {

    @Serializable
    data class Main(
        val temp: Double,
        val humidity: Int,
        @SerialName("feels_like") val feelsLike: Double,
        val pressure: Int
    )

    @Serializable
    data class Wind(
        val speed: Double = 0.0
    )
}