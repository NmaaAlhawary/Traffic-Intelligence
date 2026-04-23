package com.gub.routes

import com.gub.domain.usecase.settings.UseCaseExportCsv
import com.gub.domain.usecase.settings.UseCaseExportJson
import com.gub.domain.usecase.settings.UseCaseImportCsv
import com.gub.domain.usecase.settings.UseCaseImportJson
import io.ktor.http.ContentDisposition
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.content.PartData
import io.ktor.http.content.forEachPart
import io.ktor.http.content.streamProvider
import io.ktor.server.request.receiveMultipart
import io.ktor.server.response.header
import io.ktor.server.response.respond
import io.ktor.server.response.respondFile
import io.ktor.server.routing.Route
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.route
import org.koin.ktor.ext.inject
import java.io.File
import kotlin.getValue

fun Route.settings() {

    val exportCsv by inject<UseCaseExportCsv>()
    val exportJson by inject<UseCaseExportJson>()

    val importCsv by inject<UseCaseImportCsv>()
    val importJson by inject<UseCaseImportJson>()

    route("/api/settings") {

        post("/import-csv") {
            val multipart = call.receiveMultipart()
            var fileContent: ByteArray? = null

            multipart.forEachPart { part ->
                if (part is PartData.FileItem) {
                    fileContent = part.streamProvider().readBytes()
                }
                part.dispose()
            }

            if (fileContent != null) {
                importCsv(fileContent)
                call.respond(HttpStatusCode.OK, "CSV imported successfully")
            } else {
                call.respond(HttpStatusCode.BadRequest, "No CSV file uploaded")
            }
        }

        post("/import-json") {
            val multipart = call.receiveMultipart()
            var fileContent: ByteArray? = null

            multipart.forEachPart { part ->
                if (part is PartData.FileItem) {
                    fileContent = part.streamProvider().readBytes()
                }
                part.dispose()
            }

            if (fileContent != null) {
                importJson(fileContent)
                call.respond(HttpStatusCode.OK, "JSON imported successfully")
            } else {
                call.respond(HttpStatusCode.BadRequest, "No JSON file uploaded")
            }
        }

        get("/export-csv") {
            val zipFile = File(exportCsv())
            if (zipFile.exists()) {
                call.response.header(
                    HttpHeaders.ContentDisposition,
                    ContentDisposition.Attachment.withParameter(
                        ContentDisposition.Parameters.FileName,
                        zipFile.name
                    ).toString()
                )
                call.respondFile(zipFile)
            } else {
                call.respond(HttpStatusCode.InternalServerError, "Export failed")
            }
        }

        get("/export-json") {
            val file = File(exportJson())
            if (file.exists()) {
                call.response.header(
                    HttpHeaders.ContentDisposition,
                    ContentDisposition.Attachment.withParameter(
                        ContentDisposition.Parameters.FileName,
                        file.name
                    ).toString()
                )
                call.respondFile(file)
            } else {
                call.respond(HttpStatusCode.InternalServerError, "Export failed")
            }
        }
    }
}