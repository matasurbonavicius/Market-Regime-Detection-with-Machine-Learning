"""
Simple Market Regime Predictor

A clean, no-bullshit implementation:
1. HMM finds regimes in training data (train once)
2. Map states to bull/neutral/bear by average returns
3. Random Forest learns to predict next period's regime
4. Backtest with proper timing (no lookahead)
"""

import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from hmmlearn.hmm import GaussianHMM
from typing import Dict, Optional, Tuple
import warnings

from data_loader import load_and_prepare
from backtest import run_backtest, print_results, plot_results

warnings.filterwarnings('ignore')


class HMMWithScaler(GaussianHMM):
    """GaussianHMM with attached scaler for consistent transforms."""
    scaler: StandardScaler


def train_hmm(features: npt.NDArray[np.floating], n_states: int = 3,
              random_seed: int = 42) -> HMMWithScaler:
    """Train HMM on features. Returns fitted model."""

    print(f"\nTraining HMM with {n_states} states...")

    # Set global seed for reproducibility
    np.random.seed(random_seed)

    # Standardize features for HMM
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train multiple times, keep best
    best_model: Optional[GaussianHMM] = None
    best_score = -np.inf

    for i in range(10):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=1000,
            random_state=random_seed + i  # deterministic seeds
        )
        try:
            model.fit(features_scaled)
            score = model.score(features_scaled)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    print(f"Best HMM score: {best_score:.2f}")

    if best_model is None:
        raise RuntimeError("Failed to train HMM model")

    # Store scaler in model for later use
    best_model.scaler = scaler  # type: ignore[attr-defined]

    return best_model  # type: ignore[return-value]


def predict_hmm(model: HMMWithScaler, features: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
    """Predict states using fitted HMM."""
    features_scaled = model.scaler.transform(features)
    return model.predict(features_scaled)


def map_states_to_regimes(states: npt.NDArray[np.integer],
                          prices: npt.NDArray[np.floating]) -> Dict[int, int]:
    """
    Map HMM states to regimes based on average returns.

    State with highest avg return = bull (0)
    State with lowest avg return = bear (2)
    Middle = neutral (1)
    """

    # Calculate returns
    returns = np.diff(prices) / prices[:-1]

    # Align states with returns (state[i] corresponds to return from i to i+1)
    states_for_returns = states[:-1]

    # Calculate average return per state
    unique_states = np.unique(states)
    avg_returns: Dict[int, float] = {}

    for state in unique_states:
        mask = states_for_returns == state
        if mask.sum() > 0:
            avg_returns[int(state)] = float(returns[mask].mean())
        else:
            avg_returns[int(state)] = 0.0

    # Sort states by average return
    sorted_states = sorted(avg_returns.keys(), key=lambda x: avg_returns[x], reverse=True)

    # Map: best return -> 0 (bull), worst -> 2 (bear), middle -> 1 (neutral)
    state_map: Dict[int, int] = {}
    for i, state in enumerate(sorted_states):
        if i == 0:
            state_map[state] = 0  # bull
        elif i == len(sorted_states) - 1:
            state_map[state] = 2  # bear
        else:
            state_map[state] = 1  # neutral

    print("\nState mapping (avg returns):")
    for state in sorted_states:
        regime = {0: 'BULL', 1: 'NEUTRAL', 2: 'BEAR'}[state_map[state]]
        print(f"  HMM State {state} -> {regime} (avg return: {avg_returns[state]:.4f})")

    return state_map


def create_labels(states: npt.NDArray[np.integer],
                  state_map: Dict[int, int]) -> npt.NDArray[np.integer]:
    """
    Create training labels with proper shifting.

    Features at T should predict regime at T+1.
    So labels[i] = regime at i+1.

    This means we lose the last data point (no label for it).
    """

    # Map raw states to regime labels
    regimes = np.array([state_map[int(s)] for s in states])

    # Shift: label[i] = regime[i+1]
    # features[0:n-1] predicts labels[0:n-1] which are regimes[1:n]
    labels = regimes[1:]

    return labels


def train_classifier(features: npt.NDArray[np.floating],
                     labels: npt.NDArray[np.integer],
                     val_features: Optional[npt.NDArray[np.floating]] = None,
                     val_labels: Optional[npt.NDArray[np.integer]] = None
                     ) -> Tuple[RandomForestClassifier, StandardScaler]:
    """Train Random Forest classifier."""

    print("\nTraining Random Forest...")
    print(f"  Training samples: {len(features)}")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(features_scaled, labels)

    train_acc = clf.score(features_scaled, labels)
    print(f"  Training accuracy: {train_acc:.2%}")

    if val_features is not None and val_labels is not None:
        val_scaled = scaler.transform(val_features)
        val_acc = clf.score(val_scaled, val_labels)
        print(f"  Validation accuracy: {val_acc:.2%}")

    return clf, scaler


def main() -> None:
    """Main pipeline."""

    # --- 1. Load Data ---
    data = load_and_prepare(
        symbol="SPY",
        start="2000-01-01",
        end="2023-12-01",
        train_end="2017-01-01",
        val_end="2020-01-01"
    )

    # --- 2. Train HMM on training data ONLY ---
    hmm_model = train_hmm(data.train_features, n_states=3)

    # --- 3. Get HMM states for training data ---
    train_states = predict_hmm(hmm_model, data.train_features)

    # --- 4. Map states to regimes using training returns ---
    state_map = map_states_to_regimes(train_states, data.train_prices)

    # --- 5. Create shifted labels ---
    # Features[:-1] -> Labels (which are regimes shifted by 1)
    train_labels = create_labels(train_states, state_map)
    train_features_aligned = data.train_features[:-1]  # Drop last row (no label for it)

    # For validation: use HMM to predict states, then create labels
    val_states = predict_hmm(hmm_model, data.val_features)
    val_labels = create_labels(val_states, state_map)
    val_features_aligned = data.val_features[:-1]

    # --- 6. Train classifier ---
    clf, clf_scaler = train_classifier(
        train_features_aligned,
        train_labels,
        val_features_aligned,
        val_labels
    )

    # --- 7. Predict on test set ---
    print(f"\nPredicting on test set ({len(data.test_features)} samples)...")

    test_features_scaled = clf_scaler.transform(data.test_features)
    test_predictions = clf.predict(test_features_scaled)

    # Count predictions
    print("Predictions distribution:")
    for regime, name in [(0, 'BULL'), (1, 'NEUTRAL'), (2, 'BEAR')]:
        count = (test_predictions == regime).sum()
        print(f"  {name}: {count} ({count/len(test_predictions):.1%})")

    # --- 8. Backtest ---
    print("\nRunning backtest...")


    result = run_backtest(
        dates=data.test_dates,
        prices=data.test_prices,
        predictions=test_predictions,
    )

    print_results(result)
    plot_results(result, title="Simple Regime Strategy vs Buy & Hold (Test Period)")

    # --- 9. Show feature importance ---
    print("\nTop 10 Feature Importances:")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(min(10, len(data.feature_names))):
        idx = indices[i]
        print(f"  {data.feature_names[idx]:<20} {importances[idx]:.4f}")


if __name__ == "__main__":
    main()
