# 📊 Data Normalization & Aggregation Basics

To perform fair comparisons between players and teams using SkillCorner data, it is essential to normalize for playing time and match context. This guide covers the foundational principles used in our "Getting Started" tutorials.

---

## 1. Normalizing for Playing Time (Per 90)

Since players have varying minutes on the pitch, we normalize cumulative metrics to a **Per 90 (P90)** basis.

### At Match Level
For an individual game, use the following formula:
```python
# Formula: (metric * 90) / time_played
physical_df['hi_count_p90'] = (physical_df['hi_count_full_all'] / physical_df['minutes_full_all']) * 90
```

### At Aggregate Level (Season/Career)
When looking at averages across multiple matches, avoid averaging individual P90s. Instead, divide the average metric by the average minutes:
```python
# Formula: (AVG(metric) * 90) / AVG(time_played)
# This prevents skewing from short sub appearances
```

---

## 2. Filtering for Reliability

To ensure statistical significance and "fair" comparisons, we apply thresholds to our datasets.

*   **Minimum Games:** It is standard practice to filter for players with at least **5 games** played.
*   **Minimizing Tactical Noise:** Focus on performances of **60+ minutes**. This provides a cleaner signal of a player's true capability, decoupled from tactical shifts or short energy bursts at the end of matches.

```python
# filtering in the aggregate file
reliable_players = physical_df[physical_df['count_match'] >= 5]
```

---

## 3. Physical Performance & Ball In Play (BIP)

Advanced analysis often accounts for league intensity and "dead time" by normalizing metrics per 60 minutes of **Ball In Play (BIP)** time.

BIP is defined as the sum of **TIP** (Team In Possession) and **OTIP** (Opponent Team In Possession) minutes. This creates a **Per 60 BIP** metric.

**Formula:** `(metric_tip + metric_otip) * 60 / (minutes_tip + minutes_otip)`

```python
# Example from Tutorial Part 2
physical_df['hi_count_p60_bip'] = (
    (physical_df['hi_count_full_tip'] + physical_df['hi_count_full_otip']) * 60 /
    (physical_df['minutes_full_tip'] + physical_df['minutes_full_otip'])
)
```

---

## 4. Non-Player Aggregates

When aggregating data at a non-player level (e.g., team or position group trends), it is recommended to use **player-match level data** (available in full datasets) and only include individual performances that exceed **60 or 90 minutes** to preserve metric stability.
