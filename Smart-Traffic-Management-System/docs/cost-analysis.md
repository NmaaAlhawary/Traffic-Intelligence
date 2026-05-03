# Cost Analysis for Wadi Saqra Traffic Intelligence System

Last updated: 2026-05-03

## 1) Executive Summary

| Option | What it means | Typical monthly cost (USD) | Best for |
|---|---|---:|---|
| Local-first (current style) | Detection and analytics run on local machine/server | 0 to 80 | Hackathon demos, pilots, low budget validation |
| Cloud MVP (single site, 24/7) | One GPU instance for vision + app services in cloud | 630 to 2,150 | Early production pilot |
| Production single site (HA) | Redundant GPU/app stack, stronger ops and security | 1,550 to 5,000 | Reliability and operational continuity |
| Add Gemini for assistant/reporting | Keep vision local; use Gemini for summaries and operator assistant | +30 to +300 typical (can be higher with heavy usage) | Better UX and decision support |

## 2) Current Project Architecture and Cost Drivers

| Layer | Current implementation | Primary cost driver | Cost risk level |
|---|---|---|---|
| Camera inference | Multiple local streams with YOLO-based detection | GPU/CPU runtime | High |
| Incident aggregation | Local process merging camera signals | Compute and reliability engineering | Medium |
| Dashboard/API | Local dashboard and forecast API | Web serving + ops overhead | Medium |
| Storage | Local SQLite and logs | Storage growth and retention | Low |
| Forecasting | Local model inference/training | Compute during training windows | Medium |

## 3) Scenario-Based Monthly Cost Estimates

### 3.1 Local Lab or Demo (Mostly On-Prem)

| Cost component | Estimate (USD/month) | Notes |
|---|---:|---|
| Compute infrastructure | 0 | Existing workstation or laptop |
| Power and hardware wear | 20 to 60 | Depends on runtime hours and GPU load |
| Storage and backup media | 0 to 20 | External disk or NAS split |
| Total | 0 to 80 | Lowest-cost mode |

### 3.2 Cloud MVP (Single Site, 24/7)

| Cost component | Estimate (USD/month) | Notes |
|---|---:|---|
| GPU VM for vision inference | 500 to 1,400 | Main cost center |
| App/API VM + DB + monitoring | 80 to 300 | Dashboard, forecast API, metrics |
| Object/block storage + backups | 20 to 150 | Depends on retention policy |
| Network egress | 30 to 300 | Can spike with many live viewers |
| Total | 630 to 2,150 | Typical pilot envelope |

### 3.3 Production Single Site (High Availability)

| Cost component | Estimate (USD/month) | Notes |
|---|---:|---|
| Redundant GPU services | 1,200 to 3,500 | Failover and uptime |
| Managed DB + observability + alerting | 200 to 800 | Reliability tooling |
| Security/CDN/network/compliance overhead | 150 to 700 | Depends on policy requirements |
| Total | 1,550 to 5,000 | Production-grade baseline |

## 4) What If You Use Gemini / Google API?

## 4.1 Recommended vs Non-Recommended Usage

| Pattern | Recommendation | Why |
|---|---|---|
| Event-level summarization | Recommended | Very good value per token |
| Operator assistant chat | Recommended | Improves response speed and context |
| Incident explanation and daily reporting | Recommended | Clear productivity gain |
| Frame-by-frame cloud vision for all cameras | Not recommended | Cost scales very fast |

## 4.2 Reference Gemini Rates (Examples from current public docs)

| Model | Input price (USD per 1M tokens) | Output price (USD per 1M tokens) | Typical role |
|---|---:|---:|---|
| Gemini 2.5 Flash-Lite | 0.10 | 0.40 | High-volume, low-cost text tasks |
| Gemini 2.5 Flash | 0.30 | 2.50 | Balanced quality and speed |
| Gemini 2.5 Pro | 1.25 (<=200k context) | 10.00 (<=200k context) | Complex reasoning and high quality |

Note: Grounding features such as Google Search or Maps can add separate charges.

## 4.3 Gemini Cost Examples

### Example A: Text Assistant Only (Cost-Efficient)

Assumption: 200,000 prompts/month, each with 1,000 input tokens and 300 output tokens.

| Metric | Value |
|---|---:|
| Total input tokens | 200,000,000 |
| Total output tokens | 60,000,000 |

| Model | Input cost | Output cost | Total monthly cost |
|---|---:|---:|---:|
| Gemini 2.5 Flash | 60 | 150 | 210 |
| Gemini 2.5 Flash-Lite | 20 | 24 | 44 |

### Example B: Sending Frequent Camera Snapshots (Expensive)

Assumption: 5 cameras, 1 image/sec, 30 days.

| Metric | Value |
|---|---:|
| Images per month | 12,960,000 |
| Approx tokens/image | 1,290 |
| Approx total input tokens | 16,718,400,000 |

At this scale, even input-only pricing can reach thousands of USD per month before output/tool charges.

## 5) Recommended Cost Strategy for This Project

| Priority | Action | Expected impact |
|---|---|---|
| 1 | Keep video detection local or on dedicated GPU servers | Prevents token-based cost explosion |
| 2 | Send only structured events to Gemini, not raw frames | Major API cost reduction |
| 3 | Use Flash-Lite for routine summaries, escalate selectively | 3x to 10x cheaper than premium-only usage |
| 4 | Apply token caps and monthly budget alerts | Avoids billing surprises |
| 5 | Batch non-urgent analytics/report generation | Better unit economics |

## 6) Cost Risk Register

| Risk | Trigger | Financial impact | Mitigation |
|---|---|---|---|
| API overuse from unbounded prompts | Missing rate limits and token caps | High | Per-user quotas, request shaping, hard budget caps |
| Streaming too many feeds to cloud models | Frame-level AI in cloud | Very high | Edge inference first, event-level API only |
| Viewer growth without CDN/egress controls | Many concurrent dashboard clients | Medium to high | Caching, stream throttling, adaptive bitrate |
| Excessive retention of media/logs | Long storage windows | Medium | Tiered retention and lifecycle rules |
| Single point of failure | One GPU/server only | Operational loss, indirect cost | Add redundancy for production |

## 7) Decision Matrix

| Goal | Best architecture choice | Monthly budget expectation |
|---|---|---|
| Lowest cost validation | Keep everything local | 0 to 80 |
| Balanced pilot with smart assistant | Local vision + Gemini text layer | 700 to 2,400 |
| Production reliability | Cloud HA + selective Gemini | 1,800 to 5,500 |

## 8) Inputs Needed for Precise Budgeting

| Input parameter | Why it matters |
|---|---|
| Number of active cameras | Linear driver of inference cost |
| Effective analyzed FPS per camera | Dominates GPU requirement |
| Daily dashboard concurrent users | Drives network and serving costs |
| Prompts/day and avg token size | Drives Gemini cost |
| Grounding usage (Search/Maps) | Adds tool-specific charges |
| Data retention period | Drives storage and backup cost |

## 9) Final Recommendation

Use a hybrid architecture:

1. Keep computer vision and tracking at the edge or dedicated GPU servers.
2. Add Gemini for operator assistance, event summarization, and decision explanation.
3. Avoid frame-by-frame Gemini calls for continuous CCTV streams.
4. Start with Flash-Lite for high-volume tasks and route only hard cases to Flash or Pro.

This approach usually gives the best quality-to-cost ratio for traffic intelligence systems like this project.
