"""
Semantic relevance ranking for harvested newsletter articles.

Uses sentence-transformers with lazy model loading to score article relevance
against the user's interest profile.
"""

from __future__ import annotations

import gc
from typing import Dict, List


class RelevanceRanker:
	"""Rank articles by cosine similarity against a profile embedding."""

	def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
		self.model_name = model_name

	@staticmethod
	def _article_text(article: Dict[str, str]) -> str:
		title = article.get("title", "")
		description = article.get("description", "")
		return f"{title}. {description}".strip()

	def prune_similar_articles(
		self,
		articles: List[Dict[str, object]],
		similarity_threshold: float = 0.9,
	) -> List[Dict[str, object]]:
		"""Keep only one article from clusters with high pairwise cosine similarity."""
		if not articles:
			return []

		model = None
		torch_module = None

		try:
			import torch as torch_module
			from sentence_transformers import SentenceTransformer
			from sentence_transformers import util

			model = SentenceTransformer(self.model_name, device="cpu")
			model = model.to("cpu")

			ordered_articles = sorted(
				articles,
				key=lambda article: float(article.get("relevance_score", 0.0)),
				reverse=True,
			)
			texts = [self._article_text(article) for article in ordered_articles]

			with torch_module.no_grad():
				embeddings = model.encode(
					texts,
					convert_to_tensor=True,
					normalize_embeddings=True,
				)

			kept_indices: List[int] = []
			for index, _article in enumerate(ordered_articles):
				if not kept_indices:
					kept_indices.append(index)
					continue

				current_embedding = embeddings[index : index + 1]
				kept_embeddings = embeddings[kept_indices]
				similarities = util.cos_sim(current_embedding, kept_embeddings)[0]

				if float(similarities.max().item()) < similarity_threshold:
					kept_indices.append(index)

			return [ordered_articles[index] for index in kept_indices]
		finally:
			if model is not None:
				del model
			if torch_module is not None and torch_module.cuda.is_available():
				torch_module.cuda.empty_cache()
			gc.collect()

	def score_articles(
		self,
		interest_profile_text: str,
		unique_articles: List[Dict[str, str]],
	) -> List[Dict[str, object]]:
		"""
		Score articles using cosine similarity between profile and article text.

		Returns a copy of each article with an added relevance_score float.
		"""
		if not unique_articles:
			return []

		model = None
		torch_module = None

		try:
			import torch as torch_module
			from sentence_transformers import SentenceTransformer
			from sentence_transformers import util

			# Lazy-load inside the scoring call to minimize idle memory footprint.
			model = SentenceTransformer(self.model_name, device="cpu")
			model = model.to("cpu")

			with torch_module.no_grad():
				profile_embedding = model.encode(
					interest_profile_text,
					convert_to_tensor=True,
					normalize_embeddings=True,
				)

				article_texts = [self._article_text(article) for article in unique_articles]
				article_embeddings = model.encode(
					article_texts,
					convert_to_tensor=True,
					normalize_embeddings=True,
				)

				similarities = util.cos_sim(profile_embedding, article_embeddings)[0]

			scored_articles: List[Dict[str, object]] = []
			for article, score in zip(unique_articles, similarities):
				enriched = dict(article)
				enriched["relevance_score"] = float(score.item())
				scored_articles.append(enriched)

			return scored_articles
		finally:
			if model is not None:
				del model
			if torch_module is not None and torch_module.cuda.is_available():
				torch_module.cuda.empty_cache()
			gc.collect()
