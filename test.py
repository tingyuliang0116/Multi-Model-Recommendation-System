# src/models/popularity.py
import pandas as pd
import numpy as np
from typing import List, Dict

class PopularityRecommender:
    def __init__(self):
        self.item_popularity = None
        self.global_top_items = None
    
    def fit(self, reviews_df: pd.DataFrame, metadata_df: pd.DataFrame):
        """Train popularity-based model"""
        # Calculate popularity metrics
        item_stats = reviews_df.groupby('asin').agg({
            'overall': ['count', 'mean', 'std'],
            'reviewerID': 'nunique'
        }).reset_index()
        
        # Flatten column names
        item_stats.columns = ['asin', 'review_count', 'avg_rating', 
                             'rating_std', 'unique_users']
        
        # Fill NaN std with 0
        item_stats['rating_std'] = item_stats['rating_std'].fillna(0)
        
        # Merge with metadata
        item_stats = item_stats.merge(metadata_df[['asin', 'title']], 
                                     on='asin', how='left')
        
        # Calculate popularity score (weighted rating with review count)
        # Using Bayesian average
        C = item_stats['review_count'].quantile(0.7)  # Confidence threshold
        m = item_stats['avg_rating'].mean()  # Mean rating across all items
        
        item_stats['popularity_score'] = (
            (item_stats['review_count'] / (item_stats['review_count'] + C)) * 
            item_stats['avg_rating'] + 
            (C / (item_stats['review_count'] + C)) * m
        )
        
        # Add recency factor if timestamp available
        if 'unixReviewTime' in reviews_df.columns:
            latest_reviews = reviews_df.groupby('asin')['unixReviewTime'].max().reset_index()
            latest_reviews.columns = ['asin', 'latest_review_time']
            
            item_stats = item_stats.merge(latest_reviews, on='asin', how='left')
            
            # Normalize recency (more recent = higher score)
            max_time = item_stats['latest_review_time'].max()
            item_stats['recency_score'] = (
                item_stats['latest_review_time'] / max_time
            )
            
            # Combine popularity and recency
            item_stats['final_score'] = (
                0.8 * item_stats['popularity_score'] + 
                0.2 * item_stats['recency_score']
            )
        else:
            item_stats['final_score'] = item_stats['popularity_score']
        
        # Store results
        self.item_popularity = item_stats.sort_values('final_score', ascending=False)
        self.global_top_items = self.item_popularity.head(100)  # Top 100 items
    
    def get_popular_items(self, top_k: int = 10, 
                         min_reviews: int = 10) -> List[Dict]:
        """Get most popular items"""
        filtered_items = self.item_popularity[
            self.item_popularity['review_count'] >= min_reviews
        ]
        
        top_items = filtered_items.head(top_k)
        
        recommendations = []
        for _, item in top_items.iterrows():
            recommendations.append({
                'asin': item['asin'],
                'title': item['title'],
                'avg_rating': item['avg_rating'],
                'review_count': int(item['review_count']),
                'popularity_score': item['final_score']
            })
        
        return recommendations
    