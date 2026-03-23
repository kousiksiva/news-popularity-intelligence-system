def explain_prediction(scores):

    explanation = []

    if scores["emotion"] > 0.5:
        explanation.append("High emotional content")

    if scores["readability"] > 60:
        explanation.append("Easy to read")

    if scores["diversity"] > 0.5:
        explanation.append("Rich vocabulary")

    if scores["length"] > 50:
        explanation.append("Detailed content")

    return explanation