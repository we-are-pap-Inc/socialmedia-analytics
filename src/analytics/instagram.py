"""
Instagram-specific analytics implementation.

This module provides Instagram-specific analytics functionality
extending the base AccountAnalytics class.
"""

import logging
from collections import Counter
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import AccountAnalytics

logger = logging.getLogger(__name__)


class InstagramAnalytics(AccountAnalytics):
    """
    Instagram-specific analytics implementation.
    
    Provides additional metrics and insights specific to Instagram content.
    """
    
    def calculate_platform_specific_metrics(self) -> Dict[str, Any]:
        """
        Calculate Instagram-specific metrics.
        
        Returns
        -------
        Dict[str, Any]
            Instagram-specific metrics.
        """
        if len(self.data) == 0:
            return self._empty_platform_metrics()
        
        metrics = {
            # Content type analysis
            'content_type_distribution': self._analyze_content_types(),
            
            # Reels vs regular posts performance
            'reels_performance': self._analyze_reels_performance(),
            
            # Music usage analysis
            'music_analysis': self._analyze_music_usage(),
            
            # Caption analysis
            'caption_analysis': self._analyze_captions(),
            
            # Sponsored content analysis
            'sponsored_analysis': self._analyze_sponsored_content(),
            
            # Best performing content type
            'best_content_type': self._find_best_content_type(),
            
            # Instagram-specific engagement metrics
            'avg_comments_per_post': float(self.data['comments_count'].mean()),
            'comment_to_like_ratio': self._calculate_comment_to_like_ratio(),
            
            # Video-specific metrics
            'video_metrics': self._analyze_video_content(),
            
            # Posting time optimization
            'optimal_posting_time': self._find_optimal_posting_time()
        }
        
        return metrics
    
    def generate_insights(self) -> List[str]:
        """
        Generate Instagram-specific insights.
        
        Returns
        -------
        List[str]
            List of insights about the Instagram account.
        """
        insights = []
        
        if len(self.data) == 0:
            insights.append("No data available for generating insights")
            return insights
        
        # Get metrics
        metrics = self.calculate_metrics()
        platform_metrics = metrics.get('platform_specific', {})
        
        # Content type insights
        content_dist = platform_metrics.get('content_type_distribution', {})
        if content_dist:
            dominant_type = max(content_dist, key=content_dist.get)
            insights.append(
                f"Content is primarily {dominant_type} "
                f"({content_dist[dominant_type]:.1f}% of posts)"
            )
        
        # Engagement insights
        engagement_rate = metrics.get('engagement_rate', 0)
        if engagement_rate > 5:
            insights.append(f"High engagement rate of {engagement_rate:.2f}% indicates strong audience connection")
        elif engagement_rate < 1:
            insights.append(f"Low engagement rate of {engagement_rate:.2f}% suggests room for improvement")
        
        # Reels performance
        reels_perf = platform_metrics.get('reels_performance', {})
        if reels_perf.get('reels_engagement_multiplier', 1) > 1.5:
            insights.append(
                f"Reels perform {reels_perf['reels_engagement_multiplier']:.1f}x "
                "better than regular posts - consider creating more Reels"
            )
        
        # Consistency insights
        consistency = metrics.get('content_consistency_index', 0)
        if consistency > 0.7:
            insights.append("Excellent posting consistency helps maintain audience engagement")
        elif consistency < 0.3:
            insights.append("Irregular posting schedule may be impacting growth")
        
        # Hashtag insights
        hashtag_analysis = self.get_hashtag_analysis()
        avg_hashtags = hashtag_analysis.get('avg_hashtags_per_post', 0)
        if avg_hashtags < 5:
            insights.append(
                f"Using only {avg_hashtags:.1f} hashtags per post on average - "
                "Instagram allows up to 30"
            )
        
        # Music usage
        music_analysis = platform_metrics.get('music_analysis', {})
        if music_analysis.get('posts_with_music_pct', 0) > 50:
            insights.append(
                f"{music_analysis['posts_with_music_pct']:.1f}% of posts use music - "
                "this can boost reach through audio discovery"
            )
        
        # Growth insights
        growth_velocity = metrics.get('growth_velocity', 0)
        if growth_velocity > 5:
            insights.append(f"Strong growth velocity of {growth_velocity:.1f}% daily")
        elif growth_velocity < -2:
            insights.append("Engagement is declining - consider refreshing content strategy")
        
        # Video content insights
        video_metrics = platform_metrics.get('video_metrics', {})
        if video_metrics.get('avg_video_duration', 0) > 60:
            insights.append("Long-form video content may benefit from being split into shorter clips")
        
        # Comment ratio insights
        comment_ratio = platform_metrics.get('comment_to_like_ratio', 0)
        if comment_ratio > 0.05:
            insights.append(
                f"High comment-to-like ratio ({comment_ratio:.3f}) indicates "
                "content sparks conversation"
            )
        
        return insights
    
    def _empty_platform_metrics(self) -> Dict[str, Any]:
        """Return empty platform metrics structure."""
        return {
            'content_type_distribution': {},
            'reels_performance': {},
            'music_analysis': {},
            'caption_analysis': {},
            'sponsored_analysis': {},
            'best_content_type': None,
            'avg_comments_per_post': 0,
            'comment_to_like_ratio': 0,
            'video_metrics': {},
            'optimal_posting_time': None
        }
    
    def _analyze_content_types(self) -> Dict[str, float]:
        """Analyze distribution of content types."""
        type_counts = self.data['post_type'].value_counts()
        type_pcts = (type_counts / len(self.data)) * 100
        return type_pcts.to_dict()
    
    def _analyze_reels_performance(self) -> Dict[str, Any]:
        """Analyze Reels performance vs regular posts."""
        # Identify Reels (clips in Instagram data)
        reels_mask = self.data['raw_data'].apply(
            lambda x: x.get('productType', '') == 'clips'
        )
        
        reels = self.data[reels_mask]
        regular = self.data[~reels_mask]
        
        if len(reels) == 0 or len(regular) == 0:
            return {
                'has_reels': len(reels) > 0,
                'reels_count': len(reels),
                'reels_percentage': (len(reels) / len(self.data)) * 100 if len(self.data) > 0 else 0
            }
        
        reels_engagement = reels['engagement_count'].mean()
        regular_engagement = regular['engagement_count'].mean()
        
        multiplier = reels_engagement / regular_engagement if regular_engagement > 0 else 0
        
        return {
            'has_reels': True,
            'reels_count': len(reels),
            'reels_percentage': (len(reels) / len(self.data)) * 100,
            'avg_reels_engagement': float(reels_engagement),
            'avg_regular_engagement': float(regular_engagement),
            'reels_engagement_multiplier': float(multiplier),
            'top_performing_reel': reels.nlargest(1, 'engagement_count')['url'].values[0] if len(reels) > 0 else None
        }
    
    def _analyze_music_usage(self) -> Dict[str, Any]:
        """Analyze music usage in posts."""
        posts_with_music = self.data['has_music'].sum()
        
        # Extract music details
        music_names = []
        original_audio_count = 0
        
        for _, post in self.data.iterrows():
            music_info = post.get('music_info', {})
            if isinstance(music_info, dict) and music_info:
                if music_info.get('song_name'):
                    music_names.append(music_info['song_name'])
                if music_info.get('uses_original_audio', False):
                    original_audio_count += 1
        
        # Count most used songs
        song_counts = Counter(music_names)
        top_songs = dict(song_counts.most_common(5))
        
        return {
            'posts_with_music': posts_with_music,
            'posts_with_music_pct': (posts_with_music / len(self.data)) * 100 if len(self.data) > 0 else 0,
            'original_audio_count': original_audio_count,
            'original_audio_pct': (original_audio_count / posts_with_music) * 100 if posts_with_music > 0 else 0,
            'top_songs_used': top_songs,
            'unique_songs_count': len(set(music_names))
        }
    
    def _analyze_captions(self) -> Dict[str, Any]:
        """Analyze caption characteristics."""
        caption_lengths = self.data['caption_length']
        
        # Analyze emoji usage
        emoji_count = 0
        for caption in self.data['caption'].fillna(''):
            # Simple emoji detection (not comprehensive)
            emoji_count += sum(1 for c in caption if ord(c) > 127000)
        
        # Analyze question usage
        questions = self.data['caption'].fillna('').str.contains(r'\?').sum()
        
        return {
            'avg_caption_length': float(caption_lengths.mean()),
            'max_caption_length': int(caption_lengths.max()),
            'min_caption_length': int(caption_lengths.min()),
            'posts_with_questions': questions,
            'posts_with_questions_pct': (questions / len(self.data)) * 100,
            'avg_emoji_per_post': emoji_count / len(self.data) if len(self.data) > 0 else 0,
            'empty_captions': int((caption_lengths == 0).sum()),
            'long_captions': int((caption_lengths > 300).sum())
        }
    
    def _analyze_sponsored_content(self) -> Dict[str, Any]:
        """Analyze sponsored content performance."""
        sponsored = self.data[self.data['is_sponsored']]
        non_sponsored = self.data[~self.data['is_sponsored']]
        
        if len(sponsored) == 0:
            return {
                'has_sponsored_content': False,
                'sponsored_count': 0,
                'sponsored_percentage': 0
            }
        
        return {
            'has_sponsored_content': True,
            'sponsored_count': len(sponsored),
            'sponsored_percentage': (len(sponsored) / len(self.data)) * 100,
            'avg_sponsored_engagement': float(sponsored['engagement_count'].mean()),
            'avg_non_sponsored_engagement': float(non_sponsored['engagement_count'].mean()) if len(non_sponsored) > 0 else 0,
            'sponsored_engagement_ratio': float(
                sponsored['engagement_count'].mean() / non_sponsored['engagement_count'].mean()
            ) if len(non_sponsored) > 0 and non_sponsored['engagement_count'].mean() > 0 else 0
        }
    
    def _find_best_content_type(self) -> str:
        """Find the best performing content type."""
        type_performance = self.data.groupby('post_type')['engagement_count'].mean()
        
        if len(type_performance) == 0:
            return None
        
        return type_performance.idxmax()
    
    def _calculate_comment_to_like_ratio(self) -> float:
        """Calculate average comment to like ratio."""
        ratios = []
        for _, post in self.data.iterrows():
            if post['likes_count'] > 0:
                ratio = post['comments_count'] / post['likes_count']
                ratios.append(ratio)
        
        return float(np.mean(ratios)) if ratios else 0.0
    
    def _analyze_video_content(self) -> Dict[str, Any]:
        """Analyze video-specific metrics."""
        videos = self.data[self.data['is_video']]
        
        if len(videos) == 0:
            return {
                'has_videos': False,
                'video_count': 0,
                'video_percentage': 0
            }
        
        return {
            'has_videos': True,
            'video_count': len(videos),
            'video_percentage': (len(videos) / len(self.data)) * 100,
            'avg_video_duration': float(videos['video_duration'].mean()),
            'total_video_duration': float(videos['video_duration'].sum()),
            'avg_video_views': float(videos['views_count'].mean()),
            'avg_video_engagement': float(videos['engagement_count'].mean()),
            'shortest_video': float(videos['video_duration'].min()),
            'longest_video': float(videos['video_duration'].max())
        }
    
    def _find_optimal_posting_time(self) -> Dict[str, Any]:
        """Find optimal posting time based on engagement."""
        # Group by hour and calculate average engagement
        hourly_engagement = self.data.groupby('hour')['engagement_count'].agg(['mean', 'count'])
        
        # Only consider hours with at least 2 posts
        significant_hours = hourly_engagement[hourly_engagement['count'] >= 2]
        
        if len(significant_hours) == 0:
            return None
        
        best_hour = significant_hours['mean'].idxmax()
        
        # Group by day and find best day
        daily_engagement = self.data.groupby('day_name')['engagement_count'].mean()
        best_day = daily_engagement.idxmax() if len(daily_engagement) > 0 else None
        
        return {
            'best_hour': int(best_hour),
            'best_hour_engagement': float(significant_hours.loc[best_hour, 'mean']),
            'best_day': str(best_day) if best_day else None,
            'best_day_engagement': float(daily_engagement[best_day]) if best_day else None
        }
    
    def get_carousel_analysis(self) -> Dict[str, Any]:
        """
        Analyze carousel posts performance.
        
        Returns
        -------
        Dict[str, Any]
            Carousel post analysis.
        """
        # Identify carousel posts (multiple images)
        carousels = self.data[
            self.data['raw_data'].apply(
                lambda x: isinstance(x.get('images', []), list) and len(x.get('images', [])) > 1
            )
        ]
        
        if len(carousels) == 0:
            return {
                'has_carousels': False,
                'carousel_count': 0,
                'carousel_percentage': 0
            }
        
        non_carousels = self.data.drop(carousels.index)
        
        return {
            'has_carousels': True,
            'carousel_count': len(carousels),
            'carousel_percentage': (len(carousels) / len(self.data)) * 100,
            'avg_carousel_engagement': float(carousels['engagement_count'].mean()),
            'avg_non_carousel_engagement': float(non_carousels['engagement_count'].mean()) if len(non_carousels) > 0 else 0,
            'carousel_performance_ratio': float(
                carousels['engagement_count'].mean() / non_carousels['engagement_count'].mean()
            ) if len(non_carousels) > 0 and non_carousels['engagement_count'].mean() > 0 else 0,
            'avg_images_per_carousel': float(
                carousels['raw_data'].apply(lambda x: len(x.get('images', []))).mean()
            )
        }