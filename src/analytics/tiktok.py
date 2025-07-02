"""
TikTok-specific analytics implementation.

This module provides TikTok-specific analytics functionality
extending the base AccountAnalytics class.
"""

import logging
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import AccountAnalytics

logger = logging.getLogger(__name__)


class TikTokAnalytics(AccountAnalytics):
    """
    TikTok-specific analytics implementation.
    
    Provides additional metrics and insights specific to TikTok content.
    """
    
    def calculate_platform_specific_metrics(self) -> Dict[str, Any]:
        """
        Calculate TikTok-specific metrics.
        
        Returns
        -------
        Dict[str, Any]
            TikTok-specific metrics.
        """
        if len(self.data) == 0:
            return self._empty_platform_metrics()
        
        metrics = {
            # Share rate analysis (unique to TikTok)
            'share_metrics': self._analyze_share_performance(),
            
            # Sound/Music analysis
            'sound_analysis': self._analyze_sounds(),
            
            # Viral potential analysis
            'viral_analysis': self._analyze_viral_potential(),
            
            # TikTok-specific engagement metrics
            'avg_shares_per_post': float(self.data['shares_count'].mean()),
            'share_to_view_ratio': self._calculate_share_to_view_ratio(),
            'engagement_velocity': self._calculate_engagement_velocity(),
            
            # Video duration optimization
            'duration_analysis': self._analyze_video_durations(),
            
            # FYP (For You Page) performance indicators
            'fyp_indicators': self._analyze_fyp_performance(),
            
            # Trend participation
            'trend_participation': self._analyze_trend_participation(),
            
            # Best performing video characteristics
            'top_video_characteristics': self._analyze_top_videos(),
            
            # Time-based performance
            'temporal_performance': self._analyze_temporal_patterns()
        }
        
        return metrics
    
    def generate_insights(self) -> List[str]:
        """
        Generate TikTok-specific insights.
        
        Returns
        -------
        List[str]
            List of insights about the TikTok account.
        """
        insights = []
        
        if len(self.data) == 0:
            insights.append("No data available for generating insights")
            return insights
        
        # Get metrics
        metrics = self.calculate_metrics()
        platform_metrics = metrics.get('platform_specific', {})
        
        # Share performance insights
        share_metrics = platform_metrics.get('share_metrics', {})
        if share_metrics.get('share_rate', 0) > 0.02:
            insights.append(
                f"Excellent share rate of {share_metrics['share_rate']*100:.2f}% "
                "indicates highly shareable content"
            )
        
        # Sound usage insights
        sound_analysis = platform_metrics.get('sound_analysis', {})
        if sound_analysis.get('original_sound_percentage', 0) > 50:
            insights.append(
                f"{sound_analysis['original_sound_percentage']:.1f}% of videos use original sounds - "
                "consider using trending sounds to increase reach"
            )
        elif sound_analysis.get('trending_sound_usage', 0) > 30:
            insights.append(
                "Good use of trending sounds helps videos get discovered on FYP"
            )
        
        # Viral potential insights
        viral_analysis = platform_metrics.get('viral_analysis', {})
        if viral_analysis.get('videos_over_1m_views', 0) > 0:
            insights.append(
                f"{viral_analysis['videos_over_1m_views']} videos exceeded 1M views - "
                "analyze these for replicable elements"
            )
        
        # Engagement velocity insights
        engagement_velocity = platform_metrics.get('engagement_velocity', 0)
        if engagement_velocity > 10:
            insights.append(
                f"High engagement velocity ({engagement_velocity:.1f}%) suggests "
                "content resonates quickly with viewers"
            )
        
        # Duration insights
        duration_analysis = platform_metrics.get('duration_analysis', {})
        optimal_duration = duration_analysis.get('optimal_duration_range', {})
        if optimal_duration:
            insights.append(
                f"Videos between {optimal_duration['min']:.0f}-{optimal_duration['max']:.0f} "
                "seconds perform best for this account"
            )
        
        # FYP performance insights
        fyp_indicators = platform_metrics.get('fyp_indicators', {})
        if fyp_indicators.get('likely_fyp_percentage', 0) > 50:
            insights.append(
                f"Approximately {fyp_indicators['likely_fyp_percentage']:.1f}% of videos "
                "likely reached FYP based on view patterns"
            )
        
        # Consistency insights
        consistency = metrics.get('content_consistency_index', 0)
        if consistency > 0.7:
            insights.append("Consistent posting schedule maximizes algorithm favorability")
        elif consistency < 0.3:
            insights.append("Irregular posting may be limiting reach - TikTok rewards consistency")
        
        # Growth insights
        growth_velocity = metrics.get('growth_velocity', 0)
        if growth_velocity > 10:
            insights.append(f"Exceptional growth rate of {growth_velocity:.1f}% daily")
        
        # Trend participation
        trend_participation = platform_metrics.get('trend_participation', {})
        if trend_participation.get('hashtag_diversity', 0) < 20:
            insights.append(
                "Limited hashtag diversity - explore more trending hashtags to increase discoverability"
            )
        
        # Share to view ratio insights
        share_ratio = platform_metrics.get('share_to_view_ratio', 0)
        if share_ratio > 0.01:
            insights.append(
                f"Share-to-view ratio of {share_ratio:.3f} is above average, "
                "indicating content worth sharing"
            )
        
        return insights
    
    def _empty_platform_metrics(self) -> Dict[str, Any]:
        """Return empty platform metrics structure."""
        return {
            'share_metrics': {},
            'sound_analysis': {},
            'viral_analysis': {},
            'avg_shares_per_post': 0,
            'share_to_view_ratio': 0,
            'engagement_velocity': 0,
            'duration_analysis': {},
            'fyp_indicators': {},
            'trend_participation': {},
            'top_video_characteristics': {},
            'temporal_performance': {}
        }
    
    def _analyze_share_performance(self) -> Dict[str, Any]:
        """Analyze sharing patterns and performance."""
        total_shares = self.data['shares_count'].sum()
        total_views = self.data['views_count'].sum()
        
        # Calculate share rate
        share_rate = total_shares / total_views if total_views > 0 else 0
        
        # Find most shared videos
        top_shared = self.data.nlargest(5, 'shares_count')[['url', 'shares_count', 'caption']]
        
        # Share distribution
        high_share_videos = len(self.data[self.data['shares_count'] > self.data['shares_count'].mean() * 2])
        
        return {
            'total_shares': int(total_shares),
            'share_rate': float(share_rate),
            'avg_shares_per_video': float(self.data['shares_count'].mean()),
            'median_shares': float(self.data['shares_count'].median()),
            'high_share_videos': high_share_videos,
            'high_share_percentage': (high_share_videos / len(self.data)) * 100,
            'most_shared_video': top_shared.iloc[0]['url'] if len(top_shared) > 0 else None,
            'share_variance': float(self.data['shares_count'].var())
        }
    
    def _analyze_sounds(self) -> Dict[str, Any]:
        """Analyze sound/music usage patterns."""
        # Extract sound information
        sounds = []
        original_count = 0
        
        for _, post in self.data.iterrows():
            music_info = post.get('music_info', {})
            if isinstance(music_info, dict):
                sound_name = music_info.get('name', '')
                if sound_name:
                    sounds.append(sound_name)
                if music_info.get('is_original', False):
                    original_count += 1
        
        # Count sound usage
        sound_counts = Counter(sounds)
        unique_sounds = len(set(sounds))
        
        # Identify potential trending sounds (used multiple times)
        trending_sounds = {k: v for k, v in sound_counts.items() if v > 1}
        
        return {
            'total_unique_sounds': unique_sounds,
            'original_sound_count': original_count,
            'original_sound_percentage': (original_count / len(self.data)) * 100 if len(self.data) > 0 else 0,
            'sound_reuse_rate': 1 - (unique_sounds / len(self.data)) if len(self.data) > 0 else 0,
            'top_sounds': dict(sound_counts.most_common(5)),
            'trending_sound_usage': (len([s for s in sounds if sound_counts[s] > 1]) / len(sounds)) * 100 if sounds else 0,
            'avg_sound_popularity': np.mean(list(sound_counts.values())) if sound_counts else 0
        }
    
    def _analyze_viral_potential(self) -> Dict[str, Any]:
        """Analyze viral characteristics of content."""
        # Define viral thresholds
        viral_thresholds = {
            '10k': 10000,
            '100k': 100000,
            '1m': 1000000
        }
        
        viral_counts = {}
        for label, threshold in viral_thresholds.items():
            viral_counts[f'videos_over_{label}_views'] = len(
                self.data[self.data['views_count'] >= threshold]
            )
        
        # Calculate viral score based on view distribution
        views = self.data['views_count']
        if len(views) > 0 and views.mean() > 0:
            viral_score = (views.std() / views.mean()) * 100  # Coefficient of variation
        else:
            viral_score = 0
        
        # Identify viral velocity (engagement rate for high-view videos)
        high_view_videos = self.data[self.data['views_count'] > views.quantile(0.9)]
        if len(high_view_videos) > 0:
            viral_engagement_rate = (
                high_view_videos['engagement_count'].sum() / 
                high_view_videos['views_count'].sum()
            ) * 100
        else:
            viral_engagement_rate = 0
        
        return {
            **viral_counts,
            'viral_score': float(viral_score),
            'viral_video_percentage': (viral_counts['videos_over_10k_views'] / len(self.data)) * 100 if len(self.data) > 0 else 0,
            'viral_engagement_rate': float(viral_engagement_rate),
            'max_views': int(views.max()) if len(views) > 0 else 0,
            'view_variance': float(views.var()) if len(views) > 0 else 0,
            'potential_viral_videos': len(
                self.data[
                    (self.data['views_count'] > views.mean() * 3) & 
                    (self.data['engagement_count'] > self.data['engagement_count'].mean() * 2)
                ]
            )
        }
    
    def _calculate_share_to_view_ratio(self) -> float:
        """Calculate average share to view ratio."""
        ratios = []
        for _, post in self.data.iterrows():
            if post['views_count'] > 0:
                ratio = post['shares_count'] / post['views_count']
                ratios.append(ratio)
        
        return float(np.mean(ratios)) if ratios else 0.0
    
    def _calculate_engagement_velocity(self) -> float:
        """
        Calculate engagement velocity (engagement rate weighted by recency).
        
        Returns
        -------
        float
            Engagement velocity score.
        """
        if len(self.data) == 0:
            return 0.0
        
        # Sort by timestamp
        sorted_data = self.data.sort_values('timestamp', ascending=False).copy()
        
        # Calculate days since posted
        # Convert to timezone-naive for comparison
        now = datetime.now()
        timestamps = sorted_data['timestamp']
        if timestamps.dt.tz is not None:
            timestamps = timestamps.dt.tz_localize(None)
        sorted_data['days_since'] = (
            now - timestamps
        ).dt.total_seconds() / 86400
        
        # Weight recent posts more heavily
        sorted_data['recency_weight'] = np.exp(-sorted_data['days_since'] / 30)
        
        # Calculate weighted engagement rate
        sorted_data['weighted_engagement'] = (
            sorted_data['engagement_count'] / sorted_data['views_count'].clip(lower=1)
        ) * sorted_data['recency_weight']
        
        return float(sorted_data['weighted_engagement'].mean() * 100)
    
    def _analyze_video_durations(self) -> Dict[str, Any]:
        """Analyze video duration patterns and optimal lengths."""
        durations = self.data['video_duration']
        
        if len(durations) == 0:
            return {}
        
        # Duration buckets
        duration_buckets = {
            '0-15s': (0, 15),
            '15-30s': (15, 30),
            '30-60s': (30, 60),
            '60s+': (60, float('inf'))
        }
        
        bucket_performance = {}
        for bucket_name, (min_dur, max_dur) in duration_buckets.items():
            bucket_videos = self.data[
                (durations >= min_dur) & (durations < max_dur)
            ]
            if len(bucket_videos) > 0:
                bucket_performance[bucket_name] = {
                    'count': len(bucket_videos),
                    'avg_views': float(bucket_videos['views_count'].mean()),
                    'avg_engagement': float(bucket_videos['engagement_count'].mean())
                }
        
        # Find optimal duration range
        if bucket_performance:
            best_bucket = max(
                bucket_performance.items(),
                key=lambda x: x[1]['avg_engagement']
            )[0]
            optimal_range = duration_buckets[best_bucket]
        else:
            optimal_range = None
        
        return {
            'avg_duration': float(durations.mean()),
            'median_duration': float(durations.median()),
            'duration_distribution': bucket_performance,
            'optimal_duration_range': {
                'min': optimal_range[0],
                'max': optimal_range[1] if optimal_range[1] != float('inf') else 180
            } if optimal_range else None,
            'short_form_percentage': (len(self.data[durations <= 30]) / len(self.data)) * 100,
            'duration_consistency': float(1 / (1 + durations.std() / durations.mean())) if durations.mean() > 0 else 0
        }
    
    def _analyze_fyp_performance(self) -> Dict[str, Any]:
        """Analyze For You Page (FYP) performance indicators."""
        # FYP indicators based on view/engagement patterns
        # High view count with low follower count suggests FYP reach
        
        views = self.data['views_count']
        engagement_rates = []
        
        for _, post in self.data.iterrows():
            if post['views_count'] > 0:
                rate = post['engagement_count'] / post['views_count']
                engagement_rates.append(rate)
        
        avg_engagement_rate = np.mean(engagement_rates) if engagement_rates else 0
        
        # Videos likely on FYP (high views, good engagement)
        fyp_threshold = views.quantile(0.75) if len(views) > 0 else 0
        likely_fyp = self.data[
            (views > fyp_threshold) & 
            (self.data['engagement_count'] / self.data['views_count'].clip(lower=1) > avg_engagement_rate)
        ]
        
        # Calculate FYP score
        if len(self.data) > 0:
            fyp_score = (
                (len(likely_fyp) / len(self.data)) * 
                (views.std() / views.mean() if views.mean() > 0 else 0)
            ) * 100
        else:
            fyp_score = 0
        
        return {
            'likely_fyp_videos': len(likely_fyp),
            'likely_fyp_percentage': (len(likely_fyp) / len(self.data)) * 100 if len(self.data) > 0 else 0,
            'fyp_score': float(fyp_score),
            'avg_fyp_video_views': float(likely_fyp['views_count'].mean()) if len(likely_fyp) > 0 else 0,
            'view_distribution_cv': float(views.std() / views.mean()) if len(views) > 0 and views.mean() > 0 else 0
        }
    
    def _analyze_trend_participation(self) -> Dict[str, Any]:
        """Analyze participation in trends through hashtags."""
        all_hashtags = []
        for hashtags in self.data['hashtags']:
            if isinstance(hashtags, list):
                all_hashtags.extend(hashtags)
        
        if not all_hashtags:
            return {
                'hashtag_diversity': 0,
                'avg_hashtags_per_video': 0,
                'trend_indicators': {}
            }
        
        hashtag_counts = Counter(all_hashtags)
        
        # Common TikTok trend indicators
        trend_keywords = ['challenge', 'trend', 'viral', 'fyp', 'foryou', 'duet', 'react']
        trend_hashtags = {
            tag: count for tag, count in hashtag_counts.items()
            if any(keyword in tag.lower() for keyword in trend_keywords)
        }
        
        return {
            'hashtag_diversity': len(set(all_hashtags)),
            'avg_hashtags_per_video': float(self.data['hashtag_count'].mean()),
            'unique_hashtag_ratio': len(set(all_hashtags)) / len(all_hashtags) if all_hashtags else 0,
            'trend_hashtag_usage': (len(trend_hashtags) / len(set(all_hashtags))) * 100 if all_hashtags else 0,
            'top_trend_hashtags': dict(sorted(trend_hashtags.items(), key=lambda x: x[1], reverse=True)[:5]),
            'trend_participation_score': float(
                sum(trend_hashtags.values()) / len(self.data)
            ) if len(self.data) > 0 else 0
        }
    
    def _analyze_top_videos(self) -> Dict[str, Any]:
        """Analyze characteristics of top performing videos."""
        # Get top 10% of videos by engagement
        top_percentile = int(len(self.data) * 0.1) or 1
        top_videos = self.data.nlargest(top_percentile, 'engagement_count')
        
        if len(top_videos) == 0:
            return {}
        
        # Analyze common characteristics
        characteristics = {
            'avg_duration': float(top_videos['video_duration'].mean()),
            'avg_caption_length': float(top_videos['caption_length'].mean()),
            'avg_hashtags': float(top_videos['hashtag_count'].mean()),
            'common_posting_hour': int(top_videos['hour'].mode()[0]) if len(top_videos['hour'].mode()) > 0 else None,
            'weekend_percentage': (top_videos['is_weekend'].sum() / len(top_videos)) * 100,
            'music_usage': (top_videos['has_music'].sum() / len(top_videos)) * 100,
            'avg_views': float(top_videos['views_count'].mean()),
            'avg_engagement_rate': float(
                (top_videos['engagement_count'] / top_videos['views_count'].clip(lower=1)).mean() * 100
            )
        }
        
        # Extract common words from captions
        common_words = Counter()
        for caption in top_videos['caption'].fillna(''):
            words = caption.lower().split()
            common_words.update(word for word in words if len(word) > 3)
        
        characteristics['common_caption_words'] = dict(common_words.most_common(5))
        
        return characteristics
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze time-based performance patterns."""
        # Performance by hour
        hourly_performance = self.data.groupby('hour').agg({
            'views_count': 'mean',
            'engagement_count': 'mean',
            'shares_count': 'mean'
        })
        
        # Performance by day of week
        daily_performance = self.data.groupby('day_name').agg({
            'views_count': 'mean',
            'engagement_count': 'mean',
            'shares_count': 'mean'
        })
        
        # Find best times
        best_hour_views = hourly_performance['views_count'].idxmax() if len(hourly_performance) > 0 else None
        best_day_engagement = daily_performance['engagement_count'].idxmax() if len(daily_performance) > 0 else None
        
        return {
            'best_hour_for_views': int(best_hour_views) if best_hour_views is not None else None,
            'best_day_for_engagement': str(best_day_engagement) if best_day_engagement is not None else None,
            'hourly_view_variance': float(hourly_performance['views_count'].var()) if len(hourly_performance) > 0 else 0,
            'weekend_vs_weekday': {
                'weekend_avg_views': float(
                    self.data[self.data['is_weekend']]['views_count'].mean()
                ) if len(self.data[self.data['is_weekend']]) > 0 else 0,
                'weekday_avg_views': float(
                    self.data[~self.data['is_weekend']]['views_count'].mean()
                ) if len(self.data[~self.data['is_weekend']]) > 0 else 0
            },
            'consistency_score': float(
                1 / (1 + self.data.groupby('hour').size().std())
            ) if len(self.data.groupby('hour').size()) > 1 else 0
        }
    
    def get_duet_analysis(self) -> Dict[str, Any]:
        """
        Analyze duet performance (if data available).
        
        Returns
        -------
        Dict[str, Any]
            Duet analysis results.
        """
        # Check for duet indicators in captions
        duet_keywords = ['duet', 'react', 'response', 'reply']
        duets = self.data[
            self.data['caption'].fillna('').str.lower().str.contains(
                '|'.join(duet_keywords), regex=True
            )
        ]
        
        if len(duets) == 0:
            return {
                'has_duets': False,
                'duet_count': 0,
                'duet_percentage': 0
            }
        
        non_duets = self.data.drop(duets.index)
        
        return {
            'has_duets': True,
            'duet_count': len(duets),
            'duet_percentage': (len(duets) / len(self.data)) * 100,
            'avg_duet_views': float(duets['views_count'].mean()),
            'avg_non_duet_views': float(non_duets['views_count'].mean()) if len(non_duets) > 0 else 0,
            'duet_performance_ratio': float(
                duets['views_count'].mean() / non_duets['views_count'].mean()
            ) if len(non_duets) > 0 and non_duets['views_count'].mean() > 0 else 0,
            'top_duet': duets.nlargest(1, 'views_count')['url'].values[0] if len(duets) > 0 else None
        }