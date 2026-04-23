package com.gub.features.settings.presentation.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.gub.features.settings.presentation.AccessibilityCard
import com.gub.features.settings.presentation.DashboardPreferencesCard
import com.gub.features.settings.presentation.DisplaySettingsCard
import com.gub.features.settings.presentation.LanguageRegionCard
import com.gub.features.settings.presentation.UserProfileSettingsCard

@Composable
fun GeneralSettingsSection() {
    Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
        // User Profile Settings
//        UserProfileSettingsCard()

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            DashboardPreferencesCard(modifier = Modifier.weight(1f))
            DisplaySettingsCard(modifier = Modifier.weight(1f))
        }

//        Row(
//            modifier = Modifier.fillMaxWidth(),
//            horizontalArrangement = Arrangement.spacedBy(16.dp)
//        ) {
//            LanguageRegionCard(modifier = Modifier.weight(1f))
//            AccessibilityCard(modifier = Modifier.weight(1f))
//        }
    }
}