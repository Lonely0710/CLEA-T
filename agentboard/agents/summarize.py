"""
Trajectory Summarizer for ContextEfficientAgentV2.

This module provides trajectory summarization functionality to compress
action-observation sequences while preserving essential information.
Enhanced with cloud-edge collaboration support.
"""

from typing import List, Any, Optional, Tuple, Dict
import re
import uuid
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import cloud-edge communication components
from agentboard.communication.cloud_protocol import TrajectoryData, SummaryData, CloudMessage, MessageType

# Note: Local LLM summarization removed for simplicity
# Use cloud-edge collaboration for high-quality summaries


class TrajectorySummarizer:
    """
    Summarizes trajectories by extracting and condensing action-observation pairs.

    Enhanced with cloud-edge collaboration:
    - Local summarization for quick response
    - Cloud-based LLM evaluation for high-quality summaries
    - Intelligent cache management
    - Fallback mechanisms

    This optimizer improves upon the original implementation with:
    - Better performance through optimized string operations
    - Enhanced text processing and filtering
    - Configurable summarization parameters
    - Proper type hints and documentation
    - Cloud-edge collaboration support
    """

    def __init__(self, llm_model: Optional[Any] = None, max_summary_length: int = 800,
                 enable_cloud_collaboration: bool = False, cloud_client=None):
        """
        Initialize the trajectory summarizer.

        Args:
            llm_model: Optional LLM model for advanced summarization (deprecated)
            max_summary_length: Maximum length of generated summaries
            enable_cloud_collaboration: Whether to enable cloud-edge collaboration
            cloud_client: Cloud communication client instance
        """
        self.max_summary_length = max_summary_length
        self.enable_cloud_collaboration = enable_cloud_collaboration
        self.cloud_client = cloud_client

        # Note: Local LLM summarization removed
        # Use cloud-edge collaboration for high-quality summaries

        # Cloud collaboration settings
        self.cloud_cache = {}  # Cache for cloud summaries
        self.pending_requests = {}  # Track pending cloud requests
        self.executor = ThreadPoolExecutor(max_workers=2)  # For async operations

        # Pre-compile regex patterns for better performance
        self.action_pattern = re.compile(r'^action:\s*', re.IGNORECASE)
        self.observation_pattern = re.compile(r'^observation:\s*', re.IGNORECASE)
        self.noise_pattern = re.compile(r'^(check valid|inventory)\s*', re.IGNORECASE)

    def _extract_text_from_item(self, item: Any) -> str:
        """
        Extract meaningful text from trajectory items.

        Args:
            item: A trajectory item which could be a tuple, list, or string

        Returns:
            Extracted text content
        """
        if isinstance(item, str):
            return item.strip()

        if isinstance(item, (list, tuple)) and len(item) >= 2:
            # Handle (key, value) pairs like ("Action", text) or ("Observation", text)
            text = str(item[1]).strip()
            # Remove common prefixes
            text = self.action_pattern.sub('', text)
            text = self.observation_pattern.sub('', text)
            return text

        return str(item).strip()

    def _filter_content(self, text: str) -> bool:
        """
        Filter out noise and redundant content.

        Args:
            text: Text content to filter

        Returns:
            True if content should be kept, False otherwise
        """
        if not text or len(text.strip()) == 0:
            return False

        # Filter out noise like check valid actions, inventory commands
        if self.noise_pattern.match(text):
            return False

        # Filter out very short, non-informative content
        if len(text.strip()) < 3:
            return False

        return True

    def _condense_trajectory(self, trajectory_items: List[str]) -> str:
        """
        Condense trajectory items into a coherent summary.

        Args:
            trajectory_items: List of filtered text items from trajectory

        Returns:
            Condensed summary text
        """
        if not trajectory_items:
            return ""

        # Strategy 1: Keep first few and last few items for context
        if len(trajectory_items) <= 6:
            # For short trajectories, include everything
            condensed_items = trajectory_items
        else:
            # For longer trajectories, keep first 3 and last 3 items
            condensed_items = trajectory_items[:3] + ["..."] + trajectory_items[-3:]

        # Remove consecutive duplicates while preserving order
        unique_items = []
        prev_item = None
        for item in condensed_items:
            if item != prev_item:
                unique_items.append(item)
                prev_item = item

        return " ".join(unique_items)

    def generate_summary(self, trajectories: List[List[Any]], subgoals: List[Any]) -> List[str]:
        """
        Generate summaries for multiple trajectories with optional cloud collaboration.

        Args:
            trajectories: List of trajectories, each trajectory is a list of items
            subgoals: List of subgoals corresponding to trajectories

        Returns:
            List of summary strings
        """
        summaries: List[str] = []

        for i, trajectory in enumerate(trajectories):
            if not trajectory:
                summaries.append("")
                continue

            # Generate unique trajectory ID
            trajectory_id = str(uuid.uuid4())

            # Convert to TrajectoryData format for cloud processing
            trajectory_data = self._convert_to_trajectory_data(
                trajectory_id, trajectory, subgoals[i] if i < len(subgoals) else None
            )

            # Try cloud collaboration if enabled
            if self.enable_cloud_collaboration and self.cloud_client:
                # First check cache
                cache_key = self._get_cache_key(trajectory_data)
                if cache_key in self.cloud_cache:
                    summaries.append(self.cloud_cache[cache_key])
                    continue

                # Try to get cloud summary (non-blocking)
                cloud_summary = self._try_get_cloud_summary(trajectory_data)
                if cloud_summary:
                    summaries.append(cloud_summary)
                    # Cache the result
                    self.cloud_cache[cache_key] = cloud_summary
                    # Submit to cloud for better summary asynchronously
                    self._submit_to_cloud_async(trajectory_data)
                    continue

            # Fallback to local summarization
            summary = self._generate_local_summary(trajectory)

            # Submit to cloud for improvement if collaboration is enabled
            if self.enable_cloud_collaboration and self.cloud_client:
                self._submit_to_cloud_async(trajectory_data)

            summaries.append(summary)

        return summaries

    def _convert_to_trajectory_data(self, trajectory_id: str, trajectory: List[Any],
                                   subgoal: Any = None) -> TrajectoryData:
        """Convert trajectory to TrajectoryData format"""
        items = []
        for step in trajectory:
            if isinstance(step, list):
                # Already in format [("Action", text), ("Observation", text)]
                step_items = []
                for item in step:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        step_items.append({
                            "type": item[0],
                            "content": str(item[1]),
                            "timestamp": time.time()
                        })
                items.append(step_items)
            else:
                # Single item
                if isinstance(step, (list, tuple)) and len(step) >= 2:
                    items.append([{
                        "type": step[0],
                        "content": str(step[1]),
                        "timestamp": time.time()
                    }])

        subgoals = []
        if subgoal:
            if isinstance(subgoal, (list, tuple)) and len(subgoal) >= 2:
                subgoals.append({
                    "type": subgoal[0],
                    "content": str(subgoal[1]),
                    "timestamp": time.time()
                })

        return TrajectoryData.from_compatible_format(
            trajectory_id=trajectory_id,
            data=trajectory,
            subgoals=subgoals if subgoals else None
        )

    def _get_cache_key(self, trajectory_data: TrajectoryData) -> str:
        """Generate cache key for trajectory"""
        # Simple hash based on trajectory content
        content_str = str(len(trajectory_data.items))
        for step in trajectory_data.items[:3]:  # Use first 3 steps for caching
            for item in step:
                content_str += f"{item.type}:{item.content[:20]}"
        return str(hash(content_str))

    def _try_get_cloud_summary(self, trajectory_data: TrajectoryData) -> Optional[str]:
        """Try to get cached cloud summary"""
        cache_key = self._get_cache_key(trajectory_data)
        return self.cloud_cache.get(cache_key)

    def _submit_to_cloud_async(self, trajectory_data: TrajectoryData):
        """Submit trajectory to cloud for async processing"""
        if not self.cloud_client:
            return

        def submit_task():
            try:
                # Create and send trajectory upload message
                upload_msg = CloudMessage.create_trajectory_upload(trajectory_data)
                self.cloud_client.send_message(upload_msg)

                # Request summary
                request_msg = CloudMessage.create_summary_request(
                    trajectory_data.trajectory_id,
                    request_summary=True,
                    request_evaluation=True
                )
                self.cloud_client.send_message(request_msg)

                # Store pending request
                self.pending_requests[trajectory_data.trajectory_id] = {
                    "trajectory_data": trajectory_data,
                    "timestamp": time.time()
                }

            except Exception as e:
                print(f"[Summarizer] Cloud submission failed: {e}")

        # Submit to thread pool
        self.executor.submit(submit_task)

    def handle_cloud_response(self, summary_data: SummaryData):
        """Handle cloud summary response"""
        # Update cache
        trajectory_data = self.pending_requests.get(summary_data.trajectory_id)
        if trajectory_data:
            trajectory_data = trajectory_data["trajectory_data"]
            cache_key = self._get_cache_key(trajectory_data)
            self.cloud_cache[cache_key] = summary_data.summary

            # Remove from pending
            if summary_data.trajectory_id in self.pending_requests:
                del self.pending_requests[summary_data.trajectory_id]

            # Log the improvement
            print(f"[Summarizer] Received cloud summary for {summary_data.trajectory_id}")

    def _generate_local_summary(self, trajectory: List[Any]) -> str:
        """Generate summary locally using simple text processing"""
        # Use simple summarization without LLM
        return self._generate_simple_summary(trajectory)

    
    def _generate_simple_summary(self, trajectory: List[Any]) -> str:
        """Generate summary using simple text processing (original implementation)"""
        # Extract and filter text content efficiently
        extracted_texts = []

        for item in trajectory:
            if isinstance(item, (list, tuple)):
                # Handle nested structures like [("Action", text), ("Observation", text)]
                for sub_item in item:
                    text = self._extract_text_from_item(sub_item)
                    if self._filter_content(text):
                        extracted_texts.append(text)
            else:
                text = self._extract_text_from_item(item)
                if self._filter_content(text):
                    extracted_texts.append(text)

        # Condense the trajectory
        summary = self._condense_trajectory(extracted_texts)

        # Truncate if necessary
        if len(summary) > self.max_summary_length:
            # Smart truncation: try to end at word boundary
            truncated = summary[:self.max_summary_length - 3]
            last_space = truncated.rfind(' ')
            if last_space > self.max_summary_length * 0.8:  # Don't cut too much
                summary = truncated[:last_space] + "..."
            else:
                summary = truncated + "..."

        return summary

    def generate_single_summary(self, trajectory: List[Any]) -> str:
        """
        Generate summary for a single trajectory.

        Args:
            trajectory: Single trajectory as a list of items

        Returns:
            Summary string
        """
        summaries = self.generate_summary([trajectory], [])
        return summaries[0] if summaries else ""
