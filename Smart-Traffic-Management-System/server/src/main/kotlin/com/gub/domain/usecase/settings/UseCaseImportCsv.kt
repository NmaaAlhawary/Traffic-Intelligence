package com.gub.domain.usecase.settings

import com.gub.domain.repository.RepositorySettings

class UseCaseImportCsv(
    private val repositorySettings: RepositorySettings
) {
    suspend operator fun invoke(content: ByteArray) = repositorySettings.importCsv(content)
}