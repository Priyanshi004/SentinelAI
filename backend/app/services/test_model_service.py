
from app.services.model_service import model_service


def main():

    print("\nTesting SentinelAI model service...\n")

    model_service.load_model()

    print("\nModel loaded:", model_service.is_loaded())

    features = model_service.get_feature_columns()

    print("Expected feature count:", len(features))

    print("\nFirst 10 feature columns:")

    for feature in features[:10]:
        print("-", feature)

    print("\nMODEL SERVICE TEST PASSED")


if __name__ == "__main__":
    main()
