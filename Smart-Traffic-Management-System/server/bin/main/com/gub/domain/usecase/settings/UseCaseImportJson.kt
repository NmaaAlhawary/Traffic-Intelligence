package com.gub.domain.usecase.settings

import com.gub.domain.repository.RepositorySettings

class UseCaseImportJson(
    private val repositorySettings: RepositorySettings
) {
    suspend operator fun invoke(content: ByteArray) = repositorySettings.importJson(content)
}