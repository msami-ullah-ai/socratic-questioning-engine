from src.similarity_scorer import SimilarityScorer

def test_similarity():
    scorer = SimilarityScorer()
    s = scorer.score("Force equals mass times acceleration", 
                     "Force is mass multiplied by acceleration")
    assert s > 0.7