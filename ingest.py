import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "ds-interview-bot"

# Data Science Interview Questions and Answers
DS_INTERVIEW_DATA = [
    {
        "question": "What is the difference between supervised and unsupervised learning?",
        "answer": """Supervised Learning uses labeled data where the algorithm learns from input-output pairs to make predictions. Examples include classification (predicting categories) and regression (predicting continuous values). Common algorithms: Linear Regression, Random Forest, SVM, Neural Networks.

Unsupervised Learning works with unlabeled data to find hidden patterns or structures. The algorithm discovers relationships without predefined outputs. Examples include clustering (grouping similar data) and dimensionality reduction. Common algorithms: K-Means, DBSCAN, PCA, Hierarchical Clustering.

Key Differences:
- Supervised has labeled data; unsupervised does not
- Supervised predicts outcomes; unsupervised finds patterns
- Supervised is used for prediction; unsupervised for exploration"""
    },
    {
        "question": "Explain the bias-variance tradeoff.",
        "answer": """The bias-variance tradeoff is a fundamental concept describing the balance between two sources of error in machine learning models.

Bias is the error from oversimplified assumptions. High bias leads to underfitting - the model misses relevant patterns. Example: using linear regression for non-linear data.

Variance is the error from sensitivity to small fluctuations in training data. High variance leads to overfitting - the model captures noise as if it were a pattern. Example: a very deep decision tree.

The Tradeoff:
- Simple models: High bias, low variance (underfit)
- Complex models: Low bias, high variance (overfit)
- Goal: Find the sweet spot that minimizes total error

Solutions: Cross-validation, regularization, ensemble methods, and proper model selection help balance this tradeoff."""
    },
    {
        "question": "What is cross-validation and why is it important?",
        "answer": """Cross-validation is a technique to evaluate model performance by testing on multiple subsets of data.

K-Fold Cross-Validation (most common):
1. Split data into K equal parts (folds)
2. Train on K-1 folds, test on remaining fold
3. Repeat K times, each fold serving as test set once
4. Average the results for final performance estimate

Why it's important:
- Provides reliable performance estimate
- Uses all data for both training and testing
- Reduces overfitting risk
- Helps with model selection and hyperparameter tuning

Common variations:
- Stratified K-Fold: Maintains class distribution in each fold
- Leave-One-Out (LOO): K equals number of samples
- Time Series Split: Respects temporal order of data

Typical choice: 5-fold or 10-fold cross-validation."""
    },
    {
        "question": "What is regularization and why do we use it?",
        "answer": """Regularization is a technique to prevent overfitting by adding a penalty term to the loss function, discouraging complex models.

Types of Regularization:

L1 Regularization (Lasso):
- Adds absolute value of coefficients as penalty
- Can shrink coefficients to exactly zero
- Performs feature selection
- Formula: Loss + λ * Σ|coefficients|

L2 Regularization (Ridge):
- Adds squared coefficients as penalty
- Shrinks coefficients but rarely to zero
- Better when all features are relevant
- Formula: Loss + λ * Σ(coefficients²)

Elastic Net:
- Combines L1 and L2 regularization
- Best of both worlds

Why use regularization:
- Prevents overfitting
- Improves generalization
- Handles multicollinearity
- Enables feature selection (L1)

The λ (lambda) parameter controls regularization strength."""
    },
    {
        "question": "Explain precision, recall, and F1-score.",
        "answer": """These are classification metrics used when accuracy isn't enough, especially for imbalanced datasets.

Precision: Of all positive predictions, how many were correct?
- Formula: TP / (TP + FP)
- High precision = few false positives
- Important when false positives are costly (spam detection)

Recall (Sensitivity): Of all actual positives, how many did we find?
- Formula: TP / (TP + FN)
- High recall = few false negatives
- Important when false negatives are costly (disease detection)

F1-Score: Harmonic mean of precision and recall
- Formula: 2 * (Precision * Recall) / (Precision + Recall)
- Balances both metrics
- Useful when you need a single metric

Where:
- TP = True Positives
- FP = False Positives
- FN = False Negatives

Tradeoff: Increasing precision often decreases recall and vice versa. Choose based on business context."""
    },
    {
        "question": "What is gradient descent?",
        "answer": """Gradient descent is an optimization algorithm used to minimize the loss function by iteratively updating model parameters.

How it works:
1. Start with random parameters
2. Calculate the gradient (slope) of loss function
3. Update parameters in opposite direction of gradient
4. Repeat until convergence

Formula: θ = θ - α * ∇J(θ)
- θ = parameters
- α = learning rate
- ∇J(θ) = gradient of loss function

Types of Gradient Descent:

Batch Gradient Descent:
- Uses entire dataset for each update
- Stable but slow for large datasets

Stochastic Gradient Descent (SGD):
- Uses one sample per update
- Fast but noisy updates

Mini-Batch Gradient Descent:
- Uses small batches (32, 64, 128 samples)
- Best of both worlds
- Most commonly used

Learning Rate importance:
- Too high: overshoots minimum, may diverge
- Too low: very slow convergence
- Solution: learning rate schedulers, adaptive methods (Adam, RMSprop)"""
    },
    {
        "question": "What is the difference between bagging and boosting?",
        "answer": """Bagging and boosting are ensemble methods that combine multiple models for better performance.

Bagging (Bootstrap Aggregating):
- Trains models in parallel on random subsets (with replacement)
- Each model is independent
- Combines predictions by voting (classification) or averaging (regression)
- Reduces variance, prevents overfitting
- Example: Random Forest

Boosting:
- Trains models sequentially
- Each model focuses on errors of previous models
- Weighted combination of predictions
- Reduces bias, can overfit if not careful
- Examples: AdaBoost, Gradient Boosting, XGBoost

Key Differences:
| Aspect | Bagging | Boosting |
|--------|---------|----------|
| Training | Parallel | Sequential |
| Focus | Reduce variance | Reduce bias |
| Data sampling | Random subsets | Weighted data |
| Model weights | Equal | Based on performance |
| Overfitting risk | Lower | Higher |

When to use:
- Bagging: High variance models (deep trees)
- Boosting: High bias models, when you need best accuracy"""
    },
    {
        "question": "Explain the curse of dimensionality.",
        "answer": """The curse of dimensionality refers to problems that arise when working with high-dimensional data.

Key Problems:

1. Data Sparsity:
- As dimensions increase, data points become spread out
- Need exponentially more data to maintain density
- Example: 10 points per dimension → 10^d points for d dimensions

2. Distance Metrics Become Meaningless:
- In high dimensions, all points become nearly equidistant
- Nearest neighbor algorithms struggle
- Clustering becomes unreliable

3. Computational Cost:
- Processing time increases with dimensions
- Memory requirements grow
- Model training becomes slower

4. Overfitting Risk:
- More features than samples leads to overfitting
- Models find spurious patterns

Solutions:
- Feature Selection: Keep only relevant features
- Dimensionality Reduction: PCA, t-SNE, UMAP
- Regularization: L1/L2 penalties
- Feature Engineering: Create meaningful combined features
- Domain Knowledge: Select features based on expertise"""
    },
    {
        "question": "What is a confusion matrix?",
        "answer": """A confusion matrix is a table that visualizes classification model performance by showing predicted vs actual values.

Structure (for binary classification):
                  Predicted
                  Neg    Pos
Actual  Neg       TN     FP
        Pos       FN     TP

Components:
- True Positives (TP): Correctly predicted positive
- True Negatives (TN): Correctly predicted negative
- False Positives (FP): Incorrectly predicted positive (Type I error)
- False Negatives (FN): Incorrectly predicted negative (Type II error)

Metrics derived from confusion matrix:
- Accuracy: (TP + TN) / Total
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- Specificity: TN / (TN + FP)
- F1-Score: 2 * (Precision * Recall) / (Precision + Recall)

Why it's useful:
- Shows all types of errors
- Helps identify model weaknesses
- Essential for imbalanced datasets
- Guides threshold selection"""
    },
    {
        "question": "What is feature scaling and why is it important?",
        "answer": """Feature scaling transforms features to similar scales, improving model performance and training speed.

Types of Scaling:

1. Standardization (Z-score normalization):
- Formula: z = (x - mean) / std
- Centers data at 0, std = 1
- Works well with outliers
- Use for: SVM, logistic regression, neural networks

2. Min-Max Normalization:
- Formula: x_scaled = (x - min) / (max - min)
- Scales to range [0, 1]
- Sensitive to outliers
- Use for: neural networks, image data

3. Robust Scaling:
- Uses median and IQR
- Robust to outliers
- Formula: x_scaled = (x - median) / IQR

Why it's important:
- Algorithms using distance (KNN, SVM) need scaled features
- Gradient descent converges faster
- Prevents features with large ranges from dominating
- Required for regularization to work properly

When NOT needed:
- Tree-based models (Random Forest, XGBoost)
- Naive Bayes"""
    },
    {
        "question": "Explain the difference between classification and regression.",
        "answer": """Classification and regression are two main types of supervised learning with different output types.

Classification:
- Predicts discrete categories/labels
- Output: class membership
- Examples: spam detection, disease diagnosis, image recognition
- Algorithms: Logistic Regression, SVM, Random Forest, Neural Networks
- Metrics: Accuracy, Precision, Recall, F1-score, AUC-ROC

Types of Classification:
- Binary: two classes (yes/no, spam/not spam)
- Multi-class: more than two classes (cat/dog/bird)
- Multi-label: multiple labels per sample

Regression:
- Predicts continuous numerical values
- Output: real numbers
- Examples: house prices, temperature, stock prices
- Algorithms: Linear Regression, Ridge, Lasso, Random Forest, Neural Networks
- Metrics: MSE, RMSE, MAE, R-squared

Key Differences:
| Aspect | Classification | Regression |
|--------|---------------|------------|
| Output | Categories | Continuous |
| Evaluation | Accuracy, F1 | MSE, R² |
| Loss function | Cross-entropy | Mean squared error |
| Decision boundary | Separates classes | Fits a curve |"""
    },
    {
        "question": "What is overfitting and how do you prevent it?",
        "answer": """Overfitting occurs when a model learns training data too well, including noise, and fails to generalize to new data.

Signs of Overfitting:
- High training accuracy, low test accuracy
- Large gap between training and validation performance
- Model captures noise as patterns

Causes:
- Model too complex for the data
- Too many features relative to samples
- Training too long
- Not enough training data

Prevention Techniques:

1. More Training Data
- More data helps model learn true patterns

2. Cross-Validation
- Use k-fold CV to detect overfitting early

3. Regularization
- L1/L2 penalties discourage complex models

4. Simplify Model
- Reduce layers, trees, or features
- Use simpler algorithms

5. Early Stopping
- Stop training when validation error increases

6. Dropout (Neural Networks)
- Randomly disable neurons during training

7. Data Augmentation
- Create variations of existing data

8. Feature Selection
- Remove irrelevant or redundant features

9. Ensemble Methods
- Combine multiple models (bagging reduces variance)"""
    },
    {
        "question": "What is the ROC curve and AUC?",
        "answer": """ROC (Receiver Operating Characteristic) curve and AUC (Area Under Curve) evaluate binary classification performance across different thresholds.

ROC Curve:
- Plots True Positive Rate (TPR) vs False Positive Rate (FPR)
- TPR (Recall) = TP / (TP + FN) - y-axis
- FPR = FP / (FP + TN) - x-axis
- Shows tradeoff at different classification thresholds

Interpreting ROC:
- Top-left corner = perfect classifier
- Diagonal line = random guessing
- Below diagonal = worse than random

AUC (Area Under ROC Curve):
- Single number summarizing ROC curve
- Range: 0 to 1
- AUC = 0.5: random guessing
- AUC = 1.0: perfect classifier
- AUC > 0.8: generally good
- AUC > 0.9: excellent

Advantages:
- Threshold-independent evaluation
- Works well for imbalanced data
- Compares models easily

Limitations:
- May be misleading for highly imbalanced data
- Doesn't show actual prediction probabilities
- Consider Precision-Recall curve as alternative"""
    },
    {
        "question": "What is PCA (Principal Component Analysis)?",
        "answer": """PCA is a dimensionality reduction technique that transforms data into uncorrelated components ordered by variance explained.

How PCA Works:
1. Standardize the data
2. Compute covariance matrix
3. Calculate eigenvectors and eigenvalues
4. Sort eigenvectors by eigenvalues (variance)
5. Select top k eigenvectors (principal components)
6. Transform data to new k-dimensional space

Key Concepts:
- Principal Components: New axes that capture maximum variance
- Eigenvalues: Amount of variance explained by each PC
- Explained Variance Ratio: Percentage of total variance per PC

When to Use PCA:
- Reduce dimensions for visualization
- Speed up training with many features
- Remove multicollinearity
- Noise reduction

Choosing Number of Components:
- Keep components explaining 90-95% variance
- Use elbow method on explained variance plot
- Consider interpretability needs

Limitations:
- Assumes linear relationships
- Sensitive to scaling (always standardize first)
- Components may not be interpretable
- Information loss with fewer components

Alternatives: t-SNE, UMAP for non-linear dimensionality reduction"""
    },
    {
        "question": "What is a Random Forest and how does it work?",
        "answer": """Random Forest is an ensemble learning method that builds multiple decision trees and combines their predictions.

How it Works:
1. Bootstrap Sampling: Create random subsets of training data
2. Build Trees: Train a decision tree on each subset
3. Random Feature Selection: At each split, consider random subset of features
4. Aggregate Predictions:
   - Classification: majority voting
   - Regression: averaging

Key Hyperparameters:
- n_estimators: number of trees (more = better but slower)
- max_depth: maximum tree depth
- min_samples_split: minimum samples to split a node
- max_features: features to consider at each split

Advantages:
- Handles high-dimensional data well
- Robust to outliers and noise
- No feature scaling needed
- Provides feature importance
- Reduces overfitting compared to single tree
- Works for classification and regression

Disadvantages:
- Less interpretable than single tree
- Slower prediction than simple models
- Can overfit on noisy data
- Large memory footprint

Feature Importance:
- Measures how much each feature contributes
- Based on reduction in impurity across all trees
- Useful for feature selection"""
    },
    {
        "question": "What is gradient boosting and how does XGBoost work?",
        "answer": """Gradient Boosting builds an ensemble of weak learners sequentially, with each new model correcting errors of previous ones.

Gradient Boosting Process:
1. Start with initial prediction (often mean)
2. Calculate residuals (errors)
3. Train new model to predict residuals
4. Add new model's predictions (scaled by learning rate)
5. Repeat steps 2-4

XGBoost (Extreme Gradient Boosting):
An optimized implementation with additional features:

Key Features:
- Regularization: L1/L2 penalties prevent overfitting
- Parallel Processing: Faster training
- Tree Pruning: More efficient than traditional methods
- Built-in Cross-Validation
- Handles Missing Values automatically
- Feature Importance built-in

Important Hyperparameters:
- n_estimators: number of boosting rounds
- learning_rate (eta): shrinkage factor (0.01-0.3)
- max_depth: tree depth (3-10 typical)
- subsample: fraction of samples per tree
- colsample_bytree: fraction of features per tree
- lambda/alpha: L2/L1 regularization

Advantages:
- Often wins ML competitions
- Handles mixed data types
- Built-in regularization
- Fast and scalable

When to Use:
- Tabular/structured data
- Need high accuracy
- Have sufficient data"""
    },
    {
        "question": "What is the difference between L1 and L2 regularization?",
        "answer": """L1 and L2 regularization add penalty terms to prevent overfitting but work differently.

L1 Regularization (Lasso):
- Penalty: λ * Σ|wi| (sum of absolute weights)
- Effect: Drives some weights to exactly zero
- Result: Sparse models with feature selection
- Geometry: Diamond-shaped constraint region

Use L1 when:
- You suspect many irrelevant features
- You want automatic feature selection
- You need a simpler, interpretable model

L2 Regularization (Ridge):
- Penalty: λ * Σwi² (sum of squared weights)
- Effect: Shrinks all weights toward zero (but not to zero)
- Result: All features kept with smaller weights
- Geometry: Circular constraint region

Use L2 when:
- Most features are relevant
- You have multicollinearity
- You don't need feature selection

Comparison:
| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|-----------|------------|
| Sparsity | Yes (zeros) | No |
| Feature selection | Built-in | No |
| Multicollinearity | Picks one feature | Handles well |
| Computation | Harder (no closed form) | Easier |
| When features correlated | Arbitrary selection | Shrinks together |

Elastic Net: Combines both L1 and L2 for best of both worlds."""
    },
    {
        "question": "Explain how a neural network learns.",
        "answer": """Neural networks learn through forward propagation, loss calculation, and backpropagation with gradient descent.

Network Structure:
- Input layer: receives features
- Hidden layers: learn representations
- Output layer: produces predictions
- Neurons connected by weights

Learning Process:

1. Forward Propagation:
- Input flows through network
- Each neuron: output = activation(Σ(weights × inputs) + bias)
- Common activations: ReLU, sigmoid, tanh
- Final layer produces prediction

2. Loss Calculation:
- Compare prediction to actual value
- Classification: cross-entropy loss
- Regression: mean squared error

3. Backpropagation:
- Calculate gradient of loss with respect to each weight
- Use chain rule to propagate gradients backward
- Each weight gets a gradient indicating how to change

4. Weight Update (Gradient Descent):
- Update weights: w = w - learning_rate × gradient
- Reduces the loss
- Repeat for many iterations (epochs)

Key Concepts:
- Learning Rate: step size for updates
- Batch Size: samples processed before update
- Epochs: complete passes through training data
- Vanishing/Exploding Gradients: training challenges

Optimizers: SGD, Adam, RMSprop help converge faster and better."""
    },
    {
        "question": "What is the purpose of activation functions in neural networks?",
        "answer": """Activation functions introduce non-linearity, enabling neural networks to learn complex patterns.

Why Needed:
- Without activation, network is just linear transformation
- Multiple linear layers = single linear layer
- Non-linearity allows learning complex decision boundaries

Common Activation Functions:

ReLU (Rectified Linear Unit):
- f(x) = max(0, x)
- Pros: Fast, reduces vanishing gradient
- Cons: "Dying ReLU" - neurons can stop learning
- Most popular for hidden layers

Leaky ReLU:
- f(x) = x if x > 0, else α*x (small α like 0.01)
- Prevents dying ReLU problem

Sigmoid:
- f(x) = 1 / (1 + e^(-x))
- Output: (0, 1)
- Use: Binary classification output
- Cons: Vanishing gradient, not zero-centered

Tanh:
- f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
- Output: (-1, 1)
- Zero-centered, better than sigmoid
- Still has vanishing gradient

Softmax:
- Converts logits to probabilities (sum to 1)
- Use: Multi-class classification output

Choosing Activation:
- Hidden layers: ReLU (default), Leaky ReLU
- Binary output: Sigmoid
- Multi-class output: Softmax
- Regression output: None (linear)"""
    },
    {
        "question": "How do you handle missing data?",
        "answer": """Missing data handling depends on the amount, pattern, and mechanism of missingness.

Types of Missing Data:
- MCAR (Missing Completely at Random): missingness unrelated to data
- MAR (Missing at Random): missingness related to observed data
- MNAR (Missing Not at Random): missingness related to missing value itself

Handling Strategies:

1. Deletion:
- Listwise: remove rows with any missing values
- Pairwise: use available data for each calculation
- When: MCAR, small proportion missing (<5%)
- Risk: Lose data, potential bias

2. Simple Imputation:
- Mean/Median: replace with central tendency
- Mode: for categorical variables
- Constant: domain-specific value
- When: Quick solution, not too many missing
- Risk: Underestimates variance

3. Advanced Imputation:
- KNN Imputation: use similar samples
- MICE: Multiple Imputation by Chained Equations
- Regression: predict missing from other features
- When: MAR, need accuracy
- Better variance estimates

4. Model-Based:
- Tree-based models handle missing natively
- XGBoost, LightGBM have built-in handling

5. Indicator Method:
- Add binary column indicating missingness
- Impute with constant
- Lets model learn missingness pattern

Best Practices:
- Analyze missingness pattern first
- Try multiple strategies, compare results
- Consider domain knowledge
- Document your approach"""
    },
    {
        "question": "What is k-means clustering and how does it work?",
        "answer": """K-means is an unsupervised algorithm that partitions data into k clusters based on distance to cluster centers.

Algorithm Steps:
1. Choose k (number of clusters)
2. Initialize k centroids randomly
3. Assign each point to nearest centroid
4. Recalculate centroids as mean of assigned points
5. Repeat steps 3-4 until convergence

Convergence: When assignments stop changing or max iterations reached

Choosing K:
- Elbow Method: Plot inertia vs k, find "elbow"
- Silhouette Score: Measures cluster quality
- Domain Knowledge: Sometimes k is known

Advantages:
- Simple and fast
- Scales to large datasets
- Easy to interpret

Disadvantages:
- Must specify k in advance
- Sensitive to initialization (use k-means++)
- Assumes spherical clusters
- Sensitive to outliers
- Only finds convex clusters

K-means++ Initialization:
- Smarter centroid initialization
- Spreads initial centroids apart
- More consistent results

When to Use:
- Data has spherical clusters
- Need fast clustering
- K is known or can be estimated

Alternatives:
- DBSCAN: finds arbitrary shapes, handles outliers
- Hierarchical: no need to specify k
- Gaussian Mixture Models: soft assignments"""
    },
    {
        "question": "What is data leakage and how do you prevent it?",
        "answer": """Data leakage occurs when information from outside the training data is used to create the model, leading to overly optimistic performance estimates.

Types of Data Leakage:

1. Target Leakage:
- Features contain information about the target
- Example: Including "treatment outcome" when predicting "should treat"
- The feature wouldn't be available at prediction time

2. Train-Test Contamination:
- Test data influences training process
- Example: Fitting scaler on entire dataset before splitting
- Example: Feature selection using all data

Common Causes:
- Temporal data: using future information
- Duplicate records in train and test
- Preprocessing before splitting
- Features derived from target

Prevention Strategies:

1. Split First, Process Later:
- Split data before any preprocessing
- Fit transformers only on training data
- Apply same transformation to test data

2. Use Pipelines:
- Encapsulate preprocessing with model
- sklearn Pipeline ensures proper ordering

3. Temporal Validation:
- For time series, always test on future data
- Never shuffle time-dependent data

4. Careful Feature Engineering:
- Ask: "Would this feature be available at prediction time?"
- Remove features too correlated with target

5. Cross-Validation Properly:
- Preprocessing inside each fold
- Use sklearn's Pipeline with cross_val_score

Detection:
- Suspiciously high accuracy
- Feature importance shows unexpected features
- Performance drops dramatically in production"""
    },
    {
        "question": "What is transfer learning?",
        "answer": """Transfer learning uses knowledge from one task/domain to improve learning on a different but related task.

Why Transfer Learning:
- Limited labeled data in target domain
- Training from scratch is expensive
- Pre-trained models capture useful features

How It Works:
1. Pre-train model on large source dataset
2. Transfer learned representations
3. Fine-tune on smaller target dataset

Common Approaches:

Feature Extraction:
- Use pre-trained model as fixed feature extractor
- Only train new classifier on top
- Fast, works with small data
- Example: Use ResNet features for medical images

Fine-Tuning:
- Start with pre-trained weights
- Train entire model (or later layers) on new data
- More flexible, requires more data
- Often freeze early layers, train later layers

Domain Adaptation:
- Handle distribution shift between domains
- Techniques to align source and target distributions

Applications:

Computer Vision:
- ImageNet pre-trained models (ResNet, VGG, EfficientNet)
- Fine-tune for specific tasks

NLP:
- Pre-trained language models (BERT, GPT)
- Fine-tune for classification, QA, etc.

Best Practices:
- Similar source and target domains work better
- More target data → more fine-tuning possible
- Learning rate: smaller for pre-trained layers
- Start frozen, gradually unfreeze layers"""
    },
    {
        "question": "What is the difference between Type I and Type II errors?",
        "answer": """Type I and Type II errors are the two types of mistakes in hypothesis testing and classification.

Type I Error (False Positive):
- Rejecting null hypothesis when it's true
- Saying positive when actually negative
- Example: Healthy person diagnosed with disease
- Also called: α error, false alarm
- Controlled by: significance level (α)

Type II Error (False Negative):
- Failing to reject null hypothesis when it's false
- Saying negative when actually positive
- Example: Sick person diagnosed as healthy
- Also called: β error, miss
- Related to: statistical power (1 - β)

Relationship:
- Reducing Type I increases Type II (and vice versa)
- Cannot minimize both simultaneously
- Must balance based on context

Real-World Examples:

Medical Testing:
- Type I: Unnecessary treatment for healthy person
- Type II: Sick person goes untreated (often worse)

Spam Detection:
- Type I: Important email marked as spam
- Type II: Spam reaches inbox

Fraud Detection:
- Type I: Blocking legitimate transaction
- Type II: Missing fraudulent transaction

Choose based on which error is more costly for your specific application."""
    },
    {
        "question": "What is the difference between generative and discriminative models?",
        "answer": """Generative and discriminative models take different approaches to classification and learning.

Discriminative Models:
- Learn boundary between classes directly
- Model P(Y|X): probability of label given features
- Focus: What distinguishes classes?

Examples:
- Logistic Regression
- SVM
- Neural Networks
- Random Forest

Advantages:
- Often more accurate for classification
- Need less data
- Simpler to train
- Directly optimize for the task

Generative Models:
- Learn how data is generated
- Model P(X|Y) and P(Y): how features arise from each class
- Use Bayes' rule: P(Y|X) = P(X|Y)P(Y) / P(X)

Examples:
- Naive Bayes
- Gaussian Mixture Models
- Hidden Markov Models
- VAEs, GANs

Advantages:
- Can generate new samples
- Handle missing data naturally
- Provide more insight into data structure
- Can detect outliers

Comparison:
| Aspect | Discriminative | Generative |
|--------|---------------|------------|
| Models | P(Y|X) | P(X|Y), P(Y) |
| Goal | Classify | Understand distribution |
| Data efficiency | Better | Needs more data |
| Missing data | Struggles | Handles well |
| Generation | Cannot | Can generate samples |

Modern Deep Generative Models:
- VAE: Variational Autoencoders
- GAN: Generative Adversarial Networks
- Diffusion Models: state-of-the-art image generation"""
    },
    {
        "question": "What is A/B testing?",
        "answer": """A/B testing is a randomized experiment comparing two versions to determine which performs better on a specific metric.

Process:
1. Define hypothesis and success metric
2. Create control (A) and treatment (B) versions
3. Randomly assign users to groups
4. Collect data for sufficient duration
5. Analyze results statistically
6. Make decision based on significance

Key Components:

Sample Size Calculation:
- Based on: baseline rate, minimum detectable effect, power, significance level
- Larger effect needs fewer samples
- Typical: 80% power, 5% significance (α)

Randomization:
- Ensures groups are comparable
- Random assignment eliminates selection bias
- Use hashing for consistent user assignment

Statistical Analysis:
- t-test or z-test for continuous metrics
- Chi-square for proportions
- Check for statistical significance (p < 0.05)
- Calculate confidence intervals

Common Metrics:
- Conversion rate
- Click-through rate
- Revenue per user
- Engagement time

Best Practices:
- Run until predetermined sample size reached
- Don't peek and stop early
- Control for multiple testing
- Check for novelty effects
- Ensure adequate test duration

Pitfalls:
- Stopping too early
- Too many variants
- Wrong randomization unit
- Ignoring practical significance
- Not accounting for seasonality"""
    },
    {
        "question": "What is the difference between correlation and causation?",
        "answer": """Correlation measures association between variables; causation means one variable directly affects another.

Correlation:
- Statistical relationship between variables
- When X changes, Y tends to change
- Measured by correlation coefficient (-1 to 1)
- Does NOT imply one causes the other

Causation:
- X directly influences Y
- Changing X will change Y
- Requires more than correlation to establish

Why Correlation ≠ Causation:

1. Confounding Variables:
- Third variable causes both X and Y
- Example: Ice cream sales and drowning both increase in summer
- Confounder: temperature

2. Reverse Causation:
- Y actually causes X, not X causes Y
- Example: Does wealth cause education or education cause wealth?

3. Coincidence:
- Spurious correlations exist
- Example: Nicolas Cage films and pool drownings

Establishing Causation:

1. Randomized Controlled Trials (RCTs):
- Gold standard for causation
- Randomly assign treatment
- Control for confounders

2. Quasi-Experimental Methods:
- Instrumental variables
- Regression discontinuity
- Difference-in-differences

3. Causal Inference Frameworks:
- Propensity Score Matching
- Do-calculus (Pearl's framework)
- Potential outcomes (Rubin)

In Data Science:
- Most ML models find correlations
- Causal inference needed for decisions
- Understanding causation improves feature engineering
- Critical for making interventions"""
    },
    {
        "question": "Explain ensemble learning.",
        "answer": """Ensemble learning combines multiple models to achieve better performance than any single model.

Why Ensembles Work:
- Different models make different errors
- Combining them cancels out individual mistakes
- "Wisdom of crowds" effect

Main Ensemble Methods:

1. Bagging (Bootstrap Aggregating):
- Train models on random subsets (with replacement)
- Models train in parallel independently
- Combine by voting or averaging
- Reduces variance
- Example: Random Forest

2. Boosting:
- Train models sequentially
- Each model focuses on previous errors
- Weighted combination of models
- Reduces bias
- Examples: AdaBoost, XGBoost, LightGBM

3. Stacking:
- Train diverse base models
- Train meta-model on base model predictions
- Can combine different algorithm types
- Most flexible but complex

4. Voting:
- Hard voting: majority vote
- Soft voting: average probabilities
- Simple but effective

Key Considerations:
- Diversity: different model types/parameters
- Quality: base models should be decent
- Independence: errors should be uncorrelated

When to Use:
- Need maximum accuracy
- Have computational resources
- Competition settings
- Production systems requiring robustness

Tradeoffs:
- Higher accuracy but slower
- Less interpretable
- More complex to deploy"""
    },
    {
        "question": "What are word embeddings and how do they work?",
        "answer": """Word embeddings are dense vector representations of words that capture semantic meaning and relationships.

Why Embeddings:
- One-hot encoding: sparse, no semantic meaning
- Embeddings: dense, similar words have similar vectors

Popular Methods:

Word2Vec:
- Learns from word context
- Two architectures:
  - CBOW: predicts word from context
  - Skip-gram: predicts context from word
- Captures analogies: king - man + woman ≈ queen

GloVe (Global Vectors):
- Uses word co-occurrence statistics
- Combines global matrix factorization with local context
- Often performs similarly to Word2Vec

FastText:
- Uses subword information (character n-grams)
- Can handle out-of-vocabulary words
- Better for morphologically rich languages

Modern Contextual Embeddings:
- BERT, GPT: word meaning depends on context
- "Bank" has different embeddings in different sentences
- Much more powerful but computationally expensive

Properties of Good Embeddings:
- Similar words are close in vector space
- Capture relationships (analogy tasks)
- Useful for downstream tasks

Using Embeddings:
- Pre-trained: GloVe, Word2Vec (fast, general)
- Fine-tuned: adapt to your domain
- Learned: train with your model end-to-end

Applications:
- Text classification
- Sentiment analysis
- Machine translation
- Named entity recognition"""
    },
    {
        "question": "What is the difference between batch, mini-batch, and stochastic gradient descent?",
        "answer": """These are variants of gradient descent differing in how much data is used per parameter update.

Batch Gradient Descent:
- Uses entire dataset for each update
- One update per epoch
- Pros: Stable convergence, accurate gradients
- Cons: Very slow for large datasets, memory intensive
- Best for: Small datasets

Stochastic Gradient Descent (SGD):
- Uses single sample for each update
- N updates per epoch (N = dataset size)
- Pros: Fast updates, can escape local minima
- Cons: Noisy updates, unstable convergence
- Best for: Online learning, very large datasets

Mini-Batch Gradient Descent:
- Uses small batch (32, 64, 128, 256 samples)
- N/batch_size updates per epoch
- Pros: Balance of speed and stability, GPU efficient
- Cons: Need to tune batch size
- Best for: Most practical applications

Comparison:
| Aspect | Batch | Mini-Batch | Stochastic |
|--------|-------|------------|------------|
| Samples per update | All | 32-256 | 1 |
| Update frequency | Low | Medium | High |
| Gradient accuracy | Exact | Approximate | Noisy |
| Memory usage | High | Medium | Low |
| Convergence | Stable | Balanced | Unstable |

Typical choice: Mini-batch with size 32-128, using Adam optimizer."""
    },
    {
        "question": "What is the difference between parametric and non-parametric models?",
        "answer": """Parametric and non-parametric models differ in their assumptions about data distribution and model complexity.

Parametric Models:
- Fixed number of parameters regardless of data size
- Assume specific data distribution
- Simpler, faster to train
- May underfit if assumptions wrong

Examples:
- Linear Regression
- Logistic Regression
- Naive Bayes
- Linear SVM

Advantages:
- Computationally efficient
- Require less data
- Easier to interpret
- Less prone to overfitting

Non-Parametric Models:
- Number of parameters grows with data
- Make fewer assumptions about data
- More flexible, can capture complex patterns
- May overfit with limited data

Examples:
- K-Nearest Neighbors
- Decision Trees
- Random Forest
- Kernel SVM
- Neural Networks (often considered non-parametric)

Advantages:
- No assumptions about distribution
- Can model complex relationships
- Often more accurate

Choosing:
- Small data → Parametric (less overfitting risk)
- Large data, complex patterns → Non-parametric
- Need interpretability → Parametric"""
    }
]

def create_documents():
    """Convert interview data to LangChain documents."""
    documents = []
    for item in DS_INTERVIEW_DATA:
        content = f"Question: {item['question']}\n\nAnswer: {item['answer']}"
        doc = Document(
            page_content=content,
            metadata={"question": item['question'], "source": "knowledge_base"}
        )
        documents.append(doc)
    return documents

def process_pdf(pdf_file):
    """Process uploaded PDF and return text."""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def get_vectorstore():
    """Get or create Pinecone vectorstore."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    return vectorstore

def add_documents_to_vectorstore(texts, source_name="uploaded"):
    """Add new documents to existing vectorstore."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    documents = [Document(page_content=t, metadata={"source": source_name}) for t in texts]
    chunks = text_splitter.split_documents(documents)
    
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    
    vectorstore.add_documents(chunks)
    return len(chunks)

def ingest_data():
    """Process and store documents in Pinecone."""
    print("Creating documents from interview data...")
    documents = create_documents()
    print(f"Created {len(documents)} documents")
    
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Check if index exists, if not create it
    existing_indexes = [index.name for index in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    
    print("Creating embeddings and storing in Pinecone...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    
    print("Data ingestion complete!")
    print(f"Vector store created with {len(chunks)} chunks in Pinecone")
    return vectorstore

if __name__ == "__main__":
    ingest_data()