"""
COMP-2200: Introduction to Machine Learning
Project: Boosting with a Hinge-Loss Twist -- STARTER FILE

Name:   Ava Kirkland
Date:   4/22/2026


=== PROJECT OVERVIEW ===

In  earlier homeworks built two classifiers from scratch:

    1) A Linear SVM that minimizes the hinge loss:
           L_hinge(m) = max(0, 1 - m)

    2) AdaBoost that (implicitly) minimizes the exponential loss:
           L_exp(m)   = exp(-m)

In BOTH cases, the quantity that matters is the MARGIN of a sample:
           m_i = y_i * F(x_i)

where F(x) is the classifier's raw real-valued score:
    - For SVM:      F(x) = w . x + b
    - For AdaBoost: F(x) = sum_t alpha_t * h_t(x)

This project asks you to stitch these two ideas together. Specifically,
generalize AdaBoost to minimize a BLENDED loss:

           L(m) = lam * exp(-m)  +  (1 - lam) * max(0, 1 - m)
                   ^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^^^^^^^^^^
                   exponential loss            hinge loss

           lam = 1.0  -> vanilla AdaBoost         (your HW solution)
           lam = 0.0  -> hinge-only reweighting   (SVM-flavored boosting)
           0 < lam < 1 -> a convex combination


=== THE BIG IDEA (read this carefully before you code) ===

AdaBoost's sample re-weighting rule isn't magic. It is EXACTLY the
(magnitude of the) derivative of the exponential loss with respect
to the ensemble score F(x). Walk through it once to be sure:

    L_exp(F) = sum_i exp(-y_i * F(x_i))
    |dL_exp / dF(x_i)|  =  exp(-y_i * F(x_i))  =  w_i  (the AdaBoost weight!)


So when vanilla AdaBoost multiplies w_i = w_i * exp(-alpha * y_i * h_t(x_i))
each round, it's actually accumulating exp(-y_i * F_t(x_i)) term by term.

That observation is the key to this project: if we swap in the derivative
of a DIFFERENT loss, we get boosting against that loss instead.

    |dL_hinge / dF(x_i)| = 1 if y_i * F(x_i) < 1  else  0

In words: only samples INSIDE the margin (the "support vectors") contribute
to the next round. Sound familiar? That's the same idea from your SVM HW,
applied here to boosting.

"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import  ConfusionMatrixDisplay

import pandas as pd



# =============================================================================
# PART 1: MARGIN-BASED LOSS FUNCTIONS
# =============================================================================
#
# All three functions take a 1-D numpy array of margins `m` (one per sample)
# and return a 1-D numpy array of per-sample loss values (same length).
#
# Keep these PURE and VECTORIZED
# =============================================================================


def exponential_loss(margins):
    """
    AdaBoost's native loss, per sample.

        L_exp(m) = exp(-m)

    Shape in, shape out:
        margins : (n_samples,) array
        returns : (n_samples,) array
    """
    return np.exp(-margins)


def hinge_loss(margins):
    """
    SVM's native loss, per sample.

        L_hinge(m) = max(0, 1 - m)

    Zero when the sample is correctly classified with margin >= 1.

    Return: The hinge loss for each sample in an array
    """
    hinge_terms = np.maximum(0, 1 - margins)
    return hinge_terms


def combined_loss(margins, lam=0.5):
    """
    Convex combination of exponential loss and hinge loss.

        L(m) = lam * exp(-m) + (1 - lam) * max(0, 1 - m)

    lam = 1.0 recovers exponential loss; lam = 0.0 recovers hinge loss.
    Return: Array of blended loss values for each sample.
    """
    loss_values = (lam * exponential_loss(margins)) + ((1- lam) * hinge_loss(margins))
    return loss_values


# =============================================================================
# PART 2: LOSS-GRADIENT SAMPLE WEIGHTS
# =============================================================================
#
# This is the heart of the project. Each function below returns the UN-NORMALIZED
# sample weights that the booster will use in the NEXT round, derived as the
# magnitude of the loss gradient with respect to F(x_i).
#
# Inputs:
#     y : (n_samples,) labels in {-1, +1}
#     F : (n_samples,) current ensemble scores = sum_t alpha_t * h_t(x_i)
#
# Output:
#     w : (n_samples,) un-normalized non-negative weights
#
# The caller (BlendedBooster.fit) will normalize the returned weights so they
# sum to 1 before feeding them to the next weak learner.
# =============================================================================


def exp_loss_weights(y, F):
    """
    |dL_exp / dF| evaluated at each sample.

        w_i = exp(-y_i * F(x_i))

    This is EXACTLY the vanilla AdaBoost cumulative weight -- verify for
    yourself that the multiplicative update
            w_i = w_i * exp(-alpha_t * y_i * h_t(x_i))
    accumulates to this expression over rounds.
    """
    loss_weights = np.exp(-y * F)



    return loss_weights


def hinge_loss_weights(y, F):
    """
    |dL_hinge / dF| evaluated at each sample.

        w_i = 1 if y_i * F(x_i) < 1  else  0

    Translation: only samples currently INSIDE the margin (or misclassified)
    contribute to the next weak learner. Well-separated points are ignored.
    This is the "support vector" idea from your SVM HW, applied to boosting.

    Return: Array of if this sample was misclassified or in margin (vectorized)
    """
    margins = y * F
    return (margins < 1).astype(int)


def combined_weights(y, F, lam=0.5):
    """
    Blended reweighting rule matching the combined_loss definition.

        w_i = lam * exp(-y_i * F_i)  +  (1 - lam) * I[y_i * F_i < 1]

    This single function lets us run three very different boosters just by
    changing lam.
    """
    weights = lam * exp_loss_weights(y, F) + (1-lam) * hinge_loss_weights(y,F)
    return weights


# =============================================================================
# PART 3: DECISION STUMP 
# =============================================================================
#
# Pulled in from a DecisionStump Homework
# =============================================================================


class DecisionStump:
    """A depth-1 decision tree: one feature, one threshold, one polarity."""

    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1

    def fit(self, X, y, sample_weights):
        n_samples, n_features = X.shape
        sample_weights = sample_weights / np.sum(sample_weights)

        best_error = float("inf")
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.percentile(feature_values, np.arange(5, 100, 5))
            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[feature_values > threshold] = -1
                    else:
                        predictions[feature_values <= threshold] = -1
                    error = np.sum(sample_weights[predictions != y])
                    if error < best_error:
                        best_error = error
                        self.feature_idx = feature_idx
                        self.threshold = threshold
                        self.polarity = polarity
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        feature_values = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[feature_values > self.threshold] = -1
        else:
            predictions[feature_values <= self.threshold] = -1
        return predictions


# =============================================================================
# PART 4: BLENDED BOOSTER
# =============================================================================
#
# This is the star of the project. Structurally it looks almost identical
# to the AdaBoost classifier from your HW -- same outer loop, same alpha
# formula, same stump-fitting step. The ONE DIFFERENCE is the sample-weight
# update at the end of each round: instead of the classic multiplicative
# AdaBoost rule, we recompute weights from scratch using combined_weights().
#
# Why recompute instead of multiplicatively update?
#
#   - For pure exp loss, the multiplicative update accumulates exactly to
#     exp(-y * F), so it's equivalent.
#   - For hinge loss, the gradient is piecewise-constant (0 or 1), so there
#     is no natural "multiplicative" version. The cleanest way to support
#     ANY loss is to track the running ensemble score F directly and ask
#     the loss for its current gradient each round. So that's what we do.
# =============================================================================


class SoftBooster:
    """
    AdaBoost-style ensemble of decision stumps trained against a
    (lam * exp-loss + (1 - lam) * hinge-loss) objective.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    lam : float in [0, 1]
        Mixing coefficient. 1.0 -> vanilla AdaBoost. 0.0 -> hinge-only.
    verbose : bool
        Print per-round loss summaries.
    """

    def __init__(self, n_estimators=50, lam=0.5, verbose=True):

        self.n_estimators = n_estimators
        self.lam = lam
        self.verbose = verbose
        self.stumps = []
        self.alphas = []
        
        # Track all three losses per round so we can compare them in plots
        self.history = {"exp_loss": [], "hinge_loss": [], "combined_loss": []}
        self.weight_history = []  # Store weights at each round for visualization

    def fit(self, X, y):
        n_samples, n_features = X.shape # get rows / cols

        self.stumps = []
        self.alphas = []
        self.history = {"exp_loss": [], "hinge_loss": [], "combined_loss": []}
        self.weight_history = []

        # RUNNING ENSEMBLE SCORE for every training point
        # F[i] = sum_{t trained so far} alpha_t * h_t(x_i)
        # Starts at zero (no stumps means every score is 0)
        F = np.zeros(n_samples)

        # INITIAL SAMPLE WEIGHTS: uniform, as in vanilla AdaBoost
        weights = np.ones(n_samples) / n_samples

        for t in range(self.n_estimators):
            # -----------------------------------------------------------------
            # STEP 1: fit a weak learner on the current sample weights
            # -----------------------------------------------------------------
            # Create a DecisionStump, fit it with X, y, and the current
            # weights, then get its predictions on X


            self.weight_history.append(weights.copy())
            stump = DecisionStump()
            stump.fit(X,y,weights)

            predictions = stump.predict(X)

            # -----------------------------------------------------------------
            # STEP 2: compute the weighted error of this stump
            # -----------------------------------------------------------------
            # This is the fraction of CURRENT weight on misclassified samples.
            # Same formula as the AdaBoost HW
            #
            # Compute epsilon = sum of weights on misclassified samples.
            # Then clip to [1e-10, 1 - 1e-10] to avoid log(0) blowups

            misclassified = predictions !=y
            epsilon = np.sum(weights[misclassified])

            #handle edge cases
            epsilon = np.clip(epsilon, 1e-10, 1- 1e-10)

            # -----------------------------------------------------------------
            # STEP 3: compute alpha (the stump's vote weight)
            # -----------------------------------------------------------------
            # For simplicity we keep the classical AdaBoost alpha formula
            # even when lam < 1. A stricter treatment would line-search alpha
            # to minimize the COMBINED loss
            #
            alpha = 0.5 * np.log((1 - epsilon) / epsilon)

            # -----------------------------------------------------------------
            # STEP 4: update the running ensemble score
            # -----------------------------------------------------------------
            # F = F + alpha * predictions
            # (elementwise -- both are (n_samples,) arrays - F and predictions alpha is a scalar)
            F = F + alpha * predictions

            # -----------------------------------------------------------------
            # STEP 5: log all three losses so we can compare them later
            # -----------------------------------------------------------------
            # Compute margins = y * F, then use your loss functions
            # from Part 1 to record mean losses in self.history
            margins = y * F
            exp_l = exponential_loss(margins)
            hinge_l = hinge_loss(margins)
            combo_l = combined_loss(margins, self.lam)
            self.history["exp_loss"].append(exp_l)
            self.history["hinge_loss"].append(hinge_l)
            self.history["combined_loss"].append(combo_l)


            # Correct AdaBoost exp-loss identity test 
            # Test that the exponential loss weights are consistent with the cumulative stump predictions so far.
            F_old = F - alpha * predictions          # undo the update to get F_t
            exp_old = np.exp(-y * F_old)             # exp-loss weights before this stump
            exp_new = np.exp(-y * F)                 # exp-loss weights after this stump
            expected = exp_old * np.exp(-alpha * y * predictions)

            if not np.allclose(exp_new, expected):
                print(f"ERROR: exp-loss identity violated at round {t+1}")


            # -----------------------------------------------------------------
            # STEP 6: recompute sample weights from the CHOSEN loss
            # -----------------------------------------------------------------
            # This is what makes the booster pluggable. At lam=1 this is
            # mathematically identical to vanilla AdaBoost's reweighting;
            # at lam<1 it is something new
            #
            # Use combined_weights(y, F, self.lam), then NORMALIZE so
            # the weights sum to 1. Watch out for the degenerate case where
            # the un-normalized weights sum to 0 (can happen with lam=0
            # once every training point is outside the margin) -- in that
            # case, break out of the loop early
            new_weights = combined_weights(y,F,self.lam)
            if np.sum(new_weights) == 0:
                print('ERROR: Weights summed together had a value of 0. \nBreak Loop')
                break

            #normalize weights
            weights = (new_weights / np.sum(new_weights))
            
            self.stumps.append(stump)
            self.alphas.append(alpha)
            if self.verbose and (t+1) % 10 == 0:
                print(f'Round {t+1}: Epsilon: {epsilon:.4f}, Alpha: {alpha:.4f}')

        return self

    def decision_function(self, X):
        """
        Return the raw ensemble score F(x) = sum_t alpha_t * h_t(x).

        This is analogous to w.x + b in your SVM HW -- a real-valued score
        whose SIGN gives the predicted class and whose MAGNITUDE gives the
        confidence / margin.
        """
        n_samples = X.shape[0]

        F = np.zeros(n_samples)
        for stump, alpha in zip(self.stumps, self.alphas):
            F += alpha * stump.predict(X)
        return F

    def predict(self, X):
        """Return hard class labels in {-1, +1}."""
        F = self.decision_function(X)
        predictions = np.sign(F)
        predictions[predictions == 0] = 1
        return predictions



class AdaBoostClassifier:
    """
    AdaBoost classifier using decision stumps as weak learners.

    Attributes:
        n_estimators: int, number of boosting rounds
        stumps: list of trained DecisionStump objects
        alphas: list of model weights (one per stump)
        weight_history: list of sample weight arrays (for visualization)
    """

    def __init__(self, n_estimators=50):
        """
        Initialize the AdaBoost classifier.

        Args:
            n_estimators: number of boosting rounds (weak learners to train)
        """
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []
        self.weight_history = []  # Store weights at each round for visualization


    def fit(self, X, y):
        """
        Train the AdaBoost classifier.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,) with values in {-1, +1}

        Returns:
            self


        Steps:
        1. Initialize sample weights: w_i = 1/n for all i
        2. For t = 1 to n_estimators:
            a. Save current weights to weight_history
            b. Train a DecisionStump on (X, y) with current sample_weights
            c. Get predictions from the stump
            d. Compute weighted error: epsilon = sum of w_i where h(x_i) != y_i
            e. Compute model weight: alpha = 0.5 * ln((1 - epsilon) / epsilon)
            f. Update sample weights: w_i = w_i * exp(-alpha * y_i * h(x_i))
            g. Normalize weights so they sum to 1
            h. Store the stump and alpha

        Remember:
            - If epsilon >= 0.5, print a warning (stump is worse than random)
            - If epsilon == 0, clip alpha to avoid infinity
        """

        n_samples, n_features = X.shape   #row, col

        #clear any previous training
        self.stumps = []
        self.alphas = []
        self.weight_history = []

        #step 1: initialize weights
        weights = np.ones(n_samples) / n_samples

        #step 2: boosting loop

        for t in range(self.n_estimators): # number of stumps collecting
            self.weight_history.append(weights.copy())

            #train a stump
            stump = DecisionStump() #instantiate an instance of the class
            stump.fit(X,y,weights)
            predictions = stump.predict(X)

            misclassified = predictions !=y
            epsilon = np.sum(weights[misclassified])

            #handle edge cases
            epsilon = np.clip(epsilon, 1e-10, 1- 1e-10)

            #warn if worse than random
            if epsilon >=0.5:
                print(f'Warning:round {t+1} has an error >= 0.5 {epsilon:.4f}')

            #compute alpha
            alpha = 0.5 * np.log((1-epsilon)/epsilon)

            weights = weights * np.exp(-alpha * y * predictions)
            
            #normalize weights
            weights = weights / np.sum(weights)
            self.stumps.append(stump)
            self.alphas.append(alpha)

            #print print every 10 secs
            if (t+1) % 10 == 0:
                print(f'Round {t+1}: Error: {epsilon:.4f}, Alpha = {alpha:.4f}')
        return self

    def predict(self, X):
        """
        Make predictions using the trained ensemble.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            predictions: numpy array of shape (n_samples,) with values in {-1, +1}


        Final prediction: H(x) = sign(sum over t of alpha_t * h_t(x))
        """



        n_samples = X.shape[0] #row - number of rows

        weighted_sum = np.zeros(n_samples)

        for stump, alpha, in zip(self.stumps, self.alphas):
            weighted_sum += alpha * stump.predict(X)
        #return the sign of the weighted sum

        #if weiighted sum is <0 return -1, if above 0 then return 1
        predictions = np.sign(weighted_sum)

        #if predictions is 0 -> 1

        predictions[predictions == 0] = 1
        return predictions


#EVALUATION AND ANALYSIS
# =============================================================================


def accuracy_score(y_true, y_pred):
    """Compute classification accuracy."""
    return np.mean(y_true == y_pred)

def recall_score(y_true, y_pred, positive_class=1):
    """
    Compute recall: TP / (TP + FN)

    Recall answers: "Of all actual positives, how many did we catch?"

    Hint:
        - TP (True Positives) = predicted positive AND actually positive
        - FN (False Negatives) = predicted negative AND actually positive
    """

    TP = np.sum(
        (y_pred == positive_class) & (y_true == positive_class)
    )
    FN = np.sum(
        (y_pred != positive_class) & (y_true == positive_class)
    )
    
    return 0.0 if TP + FN == 0 else TP/(TP + FN)

def f1_score(y_true, y_pred, positive_class=1):
    """
    Compute F1 score: 2 * (precision * recall) / (precision + recall)

    F1 is the harmonic mean of precision and recall.

    Hint: Use your precision_score and recall_score functions!
    """

    p = precision_score(y_true, y_pred, positive_class)
    r = recall_score(y_true,y_pred, positive_class)

    return 0 if p +r == 0 else 2 * (p *r) / (p + r)


def precision_score(y_true, y_pred, positive_class=1):
    """
    Compute precision: TP / (TP + FP)

    Precision answers: "Of all predicted positives, how many were actually positive?"

    Hint:
        - TP (True Positives) = predicted positive AND actually positive
        - FP (False Positives) = predicted positive AND actually negative
    """

    TP = np.sum(
        (y_pred == positive_class) & (y_true == positive_class)
    )
    FP = np.sum(
        (y_pred == positive_class) & (y_true != positive_class)
    )

    #avoid divide by zero
    if TP + FP == 0:
        return 0
    return (TP)/ (TP + FP)


def plot_weight_evolution(clf, y_train, title, save_path=None, ):
    """
    Plot how sample weights evolve during AdaBoost training.

    Args:
        clf: trained AdaBoostClassifier with weight_history
        y_train: training labels (to color points by class)
        save_path: if provided, save figure to this path


    Create a plot showing:
    - X-axis: boosting round
    - Y-axis: sample weight
    - One line per sample (or a subset of samples for clarity)
    - Color by class (positive vs negative)

    Hint: You may want to only plot a subset of samples (e.g., 20-30) for clarity.
    """

    #handle edge cases 
    if len(clf.weight_history) == 0:
        print("No weight history to plot. Run fit first.")
        return
    #convert weight history to an array: shape( n_rounds, n_samples)

    weight_matrix = np.array(clf.weight_history)

    n_rounds,n_samples = weight_matrix.shape

    #select a subset of samples to plot

    n_to_plot = min(30, n_samples)

    sample_indices = np.linspace(
        0,
        n_samples -1,
        n_to_plot,
        dtype= int
    )

    plt.figure(figsize=(10,6))

    #plot class samples

    pos_plotted = False
    neg_plotted = False

    for idx in sample_indices:
        if y_train[idx] == 1:
            label = 'Survived +1' if not pos_plotted else None

            plt.plot(
                weight_matrix[:, idx],
                color='blue',
                alpha =0.4,
                label = label
            )
            pos_plotted = True
        else:
            label = 'Died -1' if not neg_plotted else None
            plt.plot(
                weight_matrix[:,idx],
                color='red',
                alpha = 0.4,
                label = label
            )
            neg_plotted = True

    plt.xlabel('Boosting Round', fontsize = 12)
    plt.ylabel('Sample Weight', fontsize = 12)
    plt.legend(loc = 'upper right')
    plt.title(title, fontsize = 14)

    plt.grid(True, alpha = 0.3)

    if save_path:
        plt.savefig(save_path, dpi = 150, bbox_inches='tight')
        print(f"Weight evolution plot saved to {save_path}\n" )
    plt.show()



def load_clean_data():
    """
    Load the Titanic DataSet and clean up the data for be ready to be input for the model

    #Cleaning: Important for model performance. There was a increase in accuracy of about 3-6% 
    # and greatly speed up the compute time from before and after cleaning the data on basic AdaBoost.

        Removing columns that have mostly or all unique values: passenderId, ticket
        Removing columns that are mostly Nan: Cabin
        Filling Nan values for columns that are mostly filled and useful: Embarked, Age

        Hot-encoding:
            For categorical columns that can easily be transformed to numerical values: Sex, Embarked

    Returns:
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,) with values in {-1, +1}
    """
    df = pd.read_csv('Titanic-Dataset.csv')




 
    print(f"First few samples: \n{df.head()}\n")
    print(f"DataSet's shape: {df.shape}\n")
    print(f"{df.shape[0]} Samples\n")

    print(f"Before Cleaning - Number of empty data values for each Column: \n{df.isnull().sum()}\n")

    #cabin has 77% missing values so drop it, inplace will modify the orginal dataframe instead of returning a copy
    df.drop(columns=['Cabin'], inplace=True) 
    #There are only 2 missing values of embarked so filled with the most common value 
    df['Embarked'] = df['Embarked'].fillna('S')
    #age is important but has 20% missing values so fill it with the median age of the their class and Sex
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    n_samples, n_features = df.shape
    print(f'Samples: {n_samples}, Features: {n_features}\n')

    

    #should do one hot encoding for specific columns
    y = df['Survived'].to_numpy()
    #Drop the values that re unique for each or most person and the target column of survival
    X_df = df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket'])
    #Check that there are no null values going to the data set for testing or training
    print(f'Nan Values for each Training Column After Data Cleaning:\n{ X_df.isnull().sum()}\n')
    samples, features = X_df.shape
    print(f'Samples: {samples}, Features: {features}\n')
    X = pd.get_dummies(
        X_df, columns = ['Embarked', 'Sex'], drop_first = True
        ).to_numpy(dtype=float)



    # Convert labels from {0, 1} to {-1, +1} for AdaBoost
    y = 2 * y - 1  # 0 -> -1, 1 -> +1

    return X, y

    
def plot_loss_history(clf, title, n_estimators, save_path=None):
    """
    Plot the history of exponential, hinge, and combined losses over boosting rounds.

    Args:
        clf: trained SoftBooster with history of losses
        title: title for the plot 
        n_estimators: total number of boosting rounds (for x-axis scaling)
        save_path: if provided, save figure to this path
        
        save_path: if provided, save figure to this path"""
    plt.close('all')

    rounds = range(1, n_estimators + 1)

    exp_loss    = clf.history['exp_loss']
    hinge_loss   = clf.history['hinge_loss']
    combo_loss   = clf.history['combined_loss']

    fig, axes = plt.subplots(1,3, figsize=(15,5), sharey=True)

    axes[0].plot(rounds, exp_loss, color='blue', alpha=0.3)
    axes[1].plot(rounds, hinge_loss, color='red', alpha=0.3)
    axes[2].plot(rounds, combo_loss, color='green', alpha=0.3)

    axes[0].set_title('Exponential Loss', fontsize=12)
    axes[1].set_title('Hinge Loss', fontsize=12)
    axes[2].set_title('Combined Loss', fontsize=12)
    for ax in axes:
        ax.set_xlabel('Boosting Round', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()


    if save_path:
        plt.savefig(save_path, dpi = 150, bbox_inches='tight')
    plt.show()
    plt.close('all')


# =============================================================================
# PART 5: MAIN
# =============================================================================


def main():

# load in data, scale, instantiate model, fit, collect metrics, etc.
    print("Titanic Dataset -- Used to Evaludate AdaBoost and SoftBoost")
    print("-"*60)

    print("\nDataset Information")
    print("-"*60)
    print('\n')
    X, y = load_clean_data()
    print(f'First 5 samples of X\n{X[:5]}\n')
    print(f'First 5 people of if they survived\n{y[:5]}\n')


    #Train and Test split
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=42)

    train_samples = X_train.shape[0]
    test_samples = X_test.shape[0]

    print(f'Train Samples: {train_samples}')
    print(f'Test Samples: {test_samples}\n')

    print("Vanilla AdaBoost")
    print("-"*60)

    # Prepare the classificer and fit of Adaboost
    clf = AdaBoostClassifier(n_estimators=50)
    clf.fit(X_train, y_train) #sequential learning

    #Predict
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    #Find the messurements for the test set
    prec = precision_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)
    rec = recall_score(y_test, test_pred)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    # Show Results for Vanilla AdaBoost
    print("\nVanilla AdaBoost Results")   
    print("-"*60)
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print("\nTest Results")
    print("-"*60)
    print(f"Accuracy: {test_acc:.4f}\n")
    print(f"Precision: {prec:.4f}\n")
    print(f"Recall: {rec:.4f}\n")
    print(f"F1 Score: {f1:.4f}\n")

    #Plots
    plot_weight_evolution(clf, y_train, title = "AdaBoost Weights Evolution", save_path=r"results\Ada_Titanic.png")
    plt.close('all')
    ConfusionMatrixDisplay.from_predictions(y_test, test_pred, display_labels=['Died (-1)', 'Survived (+1)'])
    plt.title('Confusion Matrix for AdaBoost on Titanic Dataset')
    plt.savefig(r"results\Ada_Titanic_Confusion.png", dpi=150, bbox_inches='tight')
    plt.show()
    plt.close('all')

    #SoftBoost with different lam values
    lams = [0.0,0.1,0.5,0.9,1.0]
    


    for lamb in lams:
        print(f"SoftBoost (lam = {lamb})")
        print("-"*60)
    # Prepare the classificer and fit of Adaboost
        soft_clf = SoftBooster(n_estimators=50, lam = lamb, verbose = True)      
        soft_clf.fit(X_train, y_train)
    # Predict
        train_pred_soft = soft_clf.predict(X_train)
        test_pred_soft = soft_clf.predict(X_test)
    # Find the messurements for the test set
        soft_train_acc = accuracy_score(y_train, train_pred_soft)

        soft_test_acc = accuracy_score(y_test, test_pred_soft)
        prec_soft = precision_score(y_test, test_pred_soft)
        f1_soft = f1_score(y_test, test_pred_soft)
        rec_soft = recall_score(y_test, test_pred_soft)


        #Show Results for Soft AdaBoost

        print(f"\nSoft AdaBoost Results (lam={lamb})")
        print("-"*60)
        
        print(f"\nTraining Accuracy: {soft_train_acc:.4f}")
        print("\nTest Results")
        print("-"*60)
        print(f"Accuracy: {soft_test_acc:.4f}")
        print(f"Precision: {prec_soft:.4f}\n")
        print(f"Recall: {rec_soft:.4f}\n")
        print(f"F1 Score: {f1_soft:.4f}\n")

        plot_weight_evolution(soft_clf, y_train, title = f"SoftBoost Weights Evolution (lam = {lamb})",save_path=rf"results\Soft_Ada_Titanic_{lamb}.png")
        ConfusionMatrixDisplay.from_predictions(y_test, test_pred_soft, display_labels=['Died (-1)', 'Survived (+1)'])
        plt.title(f'Confusion Matrix for SoftBoost (Lam = {lamb}) on Titanic Dataset')
        plt.savefig(rf"results\Soft_Ada_Titanic_Confusion_{lamb}.png", dpi=150, bbox_inches='tight')
        plt.show()
        plot_loss_history(soft_clf, n_estimators = 50, title = f"SoftBoost Loss History (lam = {lamb})", save_path=rf"results\Soft_Ada_Titanic_Loss_{lamb}.png")

        plt.show()

if __name__ == "__main__":
    main()
