# cs4803-7643-deep-learning-solved



**<span style='color:red'>TO GET THIS SOLUTION VISIT:</span>** https://www.ankitcodinghub.com/product/cs7643-cs4803-7643-deep-learning-solved-3/

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;105513&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS4803-7643: Deep Learning Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">
            
<div class="kksr-stars">
    
<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">
            

<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">
            

<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">
            

<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">
            

<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">
            

<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
    
<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">
            

<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">
            

<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">
            

<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">
            

<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">
            

<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>
                

<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
Instructions

1. We will be using Gradescope to collect your assignments. Please read the following instructions for submitting to Gradescope carefully!

• Each subproblem must be submitted on a separate page. When submitting to Gradescope, make sure to mark which page(s) corresponds to each problem/sub-problem

• When submitting to Gradescope, make sure to mark which page corresponds to each problem/sub-problem. ALSO remember to append your notebook PDFs to the solutions for the problem set, as described in the instructions!

• For the coding problem, please use the provided collect_submission.sh script and upload cs7643_hw4.zip to the HW4 Code assignment on Gradescope. While we will not be explicitly grading your code, you are still required to submit it. Please make sure you have saved the most recent version of your jupyter notebook before running this script.

Please read https://stats200.stanford.edu/gradescope_tips.pdf for additional information on submitting to Gradescope.

2. LATEX’d solutions are strongly encouraged (solution template available at cc.gatech.edu/classes/AY2020/cs7643_fall/assets/sol2.tex), but scanned handwritten copies are acceptable. Hard copies are not accepted.

3. We generally encourage you to collaborate with other students.

1 Recurrent Neural Networks and Transformers

1. [Vanilla RNN for parity function: 3 points]

Let us define a sequence parity function as a function that takes in a sequence of binary inputs and returns a sequence indicating the number of 1’s in the input so far; specifically, if at time t the 1’s in the input so far is odd it returns 1, and 0 if it is even. For example, given input sequence [0,1,0,1,1,0], the parity sequence is [0,1,1,0,1,1]. Let xt denote the input at time t and yt be the boolean output of this parity function at time t.

Design a vanilla recurrent neural network to implement the parity function. Your implementation should include the equations of your hidden units and the details about activations with different input at xt (0 or 1).

2. [LSTM for parity function: 5 points]

Let us recall different gates in an LSTM Network. First gate is the “forget gate layer”:

ft = σ(Wf.[ht−1,xt] + bf) (1)

where ft is the output of forget gate, Wf is the weight matrix, ht−1 is the hidden state of step t-1, xt is the current input and bt is the bias value. Next we have “input gate layer”:

it = σ(Wi.[ht−1,xt] + bi) (2) C˜t = tanh(WC.[ht−1,xt] + bC) (3)

where it decides which values we will update and C˜t are the new candidate values that could be added to the cell state. Next we have new cell state candidate values:

Ct = ft ∗ Ct−1 + it ∗ C˜t (4)

Finally, we have the output and hidden state

ot = σ(Wo.[ht−1,xt] + bo) (5)

ht = ot ∗ tanh(Ct) (6)

Design an LSTM Network for the bit parity problem mentioned in Question 1. Specifically, provide values for Wf, bf, Wi, bi, WC, bC, Wo and bo such that the cell state Ct store the parity of bit string. Please mention any assumptions you make. For this problem, you can assume below for Sigmoid and tanh function:

(

1, if x &gt; 0

σ(x) = (7)

0, otherwise

1,

 tanh(x) = 0,

−1, if x &gt; 0 x = 0

if x &lt; 0 (8)

Hint: Recall that XOR of x and y can be represented as (x ∧ y¯) ∨ (x¯ ∧ y). Think about how you can leverage this information for equation (4).

3. [When to stop in beam search: 5 points]

Beam Search is a widely-used technique for decoding the most likely sequence from sequence models. But it is difficult to decide when to stop beam search to obtain optimality because hypotheses can finish in different steps. In this question, we will develop a formal understanding of the stopping criteria in beam search.

Let x denote the input upon which we condition our sequence model. Let y denote the output sequence. Let y&lt;t be a shorthand notation for the sub-sequence (y0,y1,…,yt−1). We say that a sequence (or hypothesis as they are sometimes referred to in this literature) y is completed

(comp(y) = true), if its last token is &lt;/s&gt;, i.e.,

comp(y) = true ↔ (y|y| = &lt;/s&gt;)

in which case it will not be further expanded in beam search.

With this notation, we can write down the maximum a-posteriori inference problem as:

y∗ = argmax Y p(yt|x,y&lt;i) (9)

y

t≤|y|

s.t. comp(y) = true (10)

We use beam search to find the (approximate) best output y∗ At time t, let Bt−1 denote the beams so far. Thus, Bt−1 is a b-length list consisting of hy,si pairs, i.e., Bt−1 = , where yi is a (t−1)-length sequence (a beam) and si is its associated

score (sum of log-conditional probabilities), i.e..

Let ◦ denote a concatenation operation, i.e. y ◦ yt represents a beam expansion where y is concatenated with yt. Beam search can be then be formalized via a topb operator that selects

(quite literally) the top-b scoring items in an expanded list of beams:

b

Bt=top(11)

Let best≤t be the best completed hypothesis so far (up to step t), i.e.

∆ n o

best≤t = max s | hy,si ∈ ∪j≤tBt,comp(y) = true (12)

Notice that if there no completed beam till time t, best≤t is undefined/empty. Now, for the proof.

Assuming that best≤t is defined at time t and the current highest scoring beam in Bt (i.e. y1) scores worse than or equal to best≤t, i.e. s1 ≤ best≤i, prove that there is no need to run beam search out further. That is, prove that the current best completed beam (corresponding to best≤t) is the overall highest-probability completed beam and future steps will be no better.

4. [Exploding Gradients: 5 points]

Learning long-term dependencies in recurrent networks suffers from a particular numerical challenge – gradients propagated over many time-steps tend to either ‘vanish’ (i.e. converge to 0, frequently) or ‘explode’ (ı.e. diverge to infinity; rarely, but with more damage to the optimization). To study this problem in a simple setting, consider the following recurrence relation without any nonlinear activation function or input x:

ht = W&gt;ht−1 (13)

where W is a weight sharing matrix for recurrent relation at any time t. Let λ1,…,λn be the eigenvalues of the weight matrix W ∈ Cn×n. Its spectral radius ρ(W) is defined as:

ρ(W) = max{|λ1|,…,|λn|} (14)

Assuming the initial hidden state is h0, write the relation between hT and h0 and explain the role of the eigenvalues of W in determining the ‘vanishing’ or ‘exploding’ property as

5. [Transformer as GNN: 5 points; Extra credit for both 4803 and 7643]

Learning representations of inputs is the bedrock of all neural networks.

In recent years, Transformer models have been widely adapted to sequence modeling tasks in the vision and language domains, while Graph Neural Networks (GNNs) have been effective in constructing representations of nodes and edges in graph data. In the following questions we will explore both Transformers and GNNs, and draw some connections between them.

Background:

Let us first take a look at a graph model. We define a directed graph G = {V,E} where V is the set of all vertices and E is the set of all edges. For ∀vi ∈ V , let us define N(vi) as the set of all of vi’s neighbours with outgoing edges towards vi. vi has a state representation hti at each time step t.

The values of hti are updated in parallel, using the same snapshot of the graph at a given time step. The procedures are as follows: We first need to aggregate the incoming data Hit0 = {fji(htj)|∀j,vj ∈ N(vi)} from neighbours using the function Agg(Hit0 ). Note that the incoming data from each neighbour is a transformed version of its representation using function fji. The aggregation function Agg(Hit0 ) can be something like the summation or the mean of elements in Hit0 .

Say the initial state at time step 0 is h0i . Now let us define the update rule for hti at time step t + 1 as the following:

hti+1 = q(hti,Agg(Hit0 )) (15)

where q is a function – , where Ht = {htn|∀n,vn ∈ V }.

Now, let us take a look at Transformer models. Recall that Transformer models build features for each word as a function of the features of all the other words with an attention mechanism over them, while RNNs update features in a sequential fashion.

To represent a Transformer model’s attention mechanism, let us define a feature representation hi for word i in sentence S. We have the standard equation for the attention update at layer l as a function of the each other word j as follows:

Attention(Qlhli,Klhlj,V lhlj) (16)

= X softmax (17)

j∈S

where Ql,Kl,V l are weight matrices for “Query”, “Key”, and “Value”. Q is a matrix that contains vector representations of one word in the sentence, while K is a matrix containing representations for all the words in the sentence. V is another matrix similar to K that has representations for all words in the sentence. As a refresher for your knowledge about the Transformer model, you can refer to this paper.

Based on the above background information, answer the following questions:

(a) If the aggregation operation for Agg(Hit0 ) is the summation of representation of all adjacent vertices, rewrite the equation 15 by replacing Agg(Hit0 )) in terms of N, f, and h.

(b) Consider the directed graph G in Fig 1. The values for the vertices at time step t are as

Figure 1

follows:

(18)

The aggregation function is:

(19)

And the function f on all the edges is:

f(x) = 2x

Now, given that (20)

(21)

where W = [1,1], what is the updated value of ?

(c) Consider the graph G in question (b). We want to alter it to represent the sentence “I eat red apples” (4 word tokens) as a fully connected graph. Each vertex represents one word token, and the edges represent the relationships among the tokens. How many edges in all would graph G contain? Notice that the edges are directed and a bi-directional edge counts as two edges.

(d) Using equations 15 and 17, show that the Transformer model’s single-head attention mechanism is equivalent to a special case of a GNN.

(e) An ongoing area of research in Transformer models for NLP is the challenge of learning very-long-term dependencies among entities in a sequence. Based on this connection with GNNs, why do you think this could be problematic?

2 Paper Review [Extra credit for 4803, regular credit for 7643]

For this homework’s paper review section, we turn to the interesting and increasingly important field of Explainable AI.

The following paper, presented at CVPR 2018, introduces a unique multi-modal approach to explainability by joint textual rationale generation and attention visualization – in the tasks of visual question answering (VQA) and activity recognition. This was a major improvement over pre-existing unimodal explainability techniques (such as the ones covered in the coding section of HW3), with the results showing complementary explanatory strengths from the visual and textual explanations. The paper can be viewed here. The evaluation rubric for this section is as follows:

6. [2 points] Briefly summarize the key contributions, strengths and weaknesses of this paper.

7. [2 points] What is your personal takeaway from this paper? This could be expressed either in terms of relating the approaches adopted in this paper to your traditional understanding of explainability techniques, or potential future directions of research in the area which the authors haven’t addressed, or anything else that struck you as being noteworthy.

3 Coding: Sequence models for image captioning and Transformer models for text classification [16.5 regular points + 1.5 extra credit points for both CS4803 and CS7643]

The coding part of this assignment will consist of implementation of sequence models for captioning images. To get started, go to https://www.cc.gatech.edu/classes/AY2020/cs7643_fall/

8cc2zXixdvAocq4v4upA9A/hw4/
