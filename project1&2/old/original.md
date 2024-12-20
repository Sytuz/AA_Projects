  In recent years, there have been breakthroughs on finding better algorithms to solve NP-hard problems, some of which have reached quasi-polynomial times. In this work, I analyze the Maximum Weighted Independent Set (MWIS) problem and developed algorithms, both exhaustive and greedy, to find solutions that are fast, scalable and precise.

  We can start this analysis by determining the average degree of a node. If we consider \(n\) as the number of nodes in a given graph and let \(k\) be the edge probability, where each pair of nodes has an edge with probability \(k\), then each node as an average degree \(d = p(n-1)\).

Now, consider that \(S \subseteq V\) is an independent set in a random graph \(G = (V,E)\). Each node in \(S\) has, on average, \(d\) neighbors that would need to be excluded from \(S\) to maintain independence. Consequently, each of those nodes can represent an exclusion zone of about \(d+1\) nodes (itself and its neighbors). Thus, we can approximate the maximum expected size \(|S|\) of an independent set in \(G\) as the total number of nodes divided by the size of each zone:

\[
|S| \approx \frac{n}{d+1} = \frac{n}{k(n-1) + 1}
\]

Although this formula doesn't take into account the weight of each node, we can apply it to our problem and deduce that for higher values of \(k\) the size of the solution subset is smaller.