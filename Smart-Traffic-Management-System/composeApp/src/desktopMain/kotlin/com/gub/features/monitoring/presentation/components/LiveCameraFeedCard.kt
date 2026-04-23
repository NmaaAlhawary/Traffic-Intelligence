package com.gub.features.monitoring.presentation.components

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Fullscreen
import androidx.compose.material.icons.filled.FullscreenExit
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun LiveCameraFeedCard(
    modifier: Modifier = Modifier,
    isExpanded: Boolean = false,
    onToggleExpand: () -> Unit = {},
    currentFrame: ImageBitmap?,
    vehicleCount: String,
    frameError: String?
) {

    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    if (isExpanded) "Camera Feed - Expanded View" else "Camera Feed",
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold
                )
                IconButton(
                    onClick = onToggleExpand,
                    modifier = Modifier.size(if (isExpanded) 32.dp else 24.dp)
                ) {
                    Icon(
                        if (isExpanded) Icons.Default.FullscreenExit else Icons.Default.Fullscreen,
                        contentDescription = if (isExpanded) "Collapse" else "Expand",
                        tint = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.size(if (isExpanded) 20.dp else 16.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(if (isExpanded) 500.dp else 200.dp)
                    .background(Color(0xFF0D1117), RoundedCornerShape(8.dp))
                    .clickable { onToggleExpand() }
            ) {
                when {
                    currentFrame != null -> {
                        Image(
                            bitmap = currentFrame!!,
                            contentDescription = "Live camera feed",
                            modifier = Modifier.fillMaxSize(),
                            contentScale = if (isExpanded) ContentScale.Fit else ContentScale.Crop
                        )
                        Box(
                            modifier = Modifier
                                .align(Alignment.TopEnd)
                                .padding(if (isExpanded) 16.dp else 8.dp)
                                .background(
                                    Color.Black.copy(alpha = 0.7f),
                                    RoundedCornerShape(6.dp)
                                )
                                .padding(
                                    horizontal = if (isExpanded) 12.dp else 8.dp,
                                    vertical = if (isExpanded) 8.dp else 4.dp
                                )
                        ) {
                            Text(
                                "Vehicles: $vehicleCount",
                                color = Color.White,
                                fontSize = if (isExpanded) 14.sp else 10.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                    frameError != null -> {
                        Text(
                            "Error: $frameError",
                            color = Color.Red,
                            modifier = Modifier.align(Alignment.Center)
                        )
                    }
                    else -> {
                        CircularProgressIndicator(
                            color = Color(0xFF4CAF50),
                            modifier = Modifier
                                .size(if (isExpanded) 48.dp else 24.dp)
                                .align(Alignment.Center)
                        )
                    }
                }
            }
        }
    }
}