### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# ╔═╡ 2848a983-6b48-4d81-b451-d3290f931426
begin
	using PlutoUI
	using Plots
	using LinearAlgebra
	using StaticArrays
	using StatsBase
	using Markdown
end

# ╔═╡ 8b2869aa-9315-4aac-a2a4-18c33be82cae
function svectors(x::AbstractMatrix{T}, ::Val{N}) where {T,N}
    """ Converts matrices to vectors of SVectors"""
    size(x,1) == N || error("sizes mismatch")
    isbitstype(T) || error("use for bitstypes only")
    reinterpret(SVector{N,T}, vec(x))
end;

# ╔═╡ 16d76581-01d3-40ab-a72a-eb2543ba9859
logrange(x1, x2, n) = (10^y for y in range(log10(x1), log10(x2), length=n));

# ╔═╡ 1fa3fa02-50fb-40d1-b7fa-9ae52efa3d16
macro timeit(expr)
	prefixes = ("m", "μ", "n")
    quote
	    local t0 = time_ns()
	    local val = $(esc(expr))
	    local t1 = time_ns()
	    local t = t1-t0
	    local index = 3 - Int(floor(log10(t)) ÷ 3) 
	    "Time elapsed = " * if index < 1
	    	t = t/1e9
	        string(round(t, sigdigits=3)) * " s" 
	    else
	        t = t * exp10(3 * (index-3))
	        string(round(t, sigdigits=3)) * " " * $prefixes[index] * "s"
		end |> Markdown.parse
    end
end;

# ╔═╡ 5d3095fe-72d2-43ef-9a63-7f67518b85eb
md"""
# Return to the origin


For a *finite* random walk of (large) length $n$, it is known that the expected number of returns to the origin $T_n$ scales like follows:

$$\left\langle T_n \right\rangle \sim \left\{
\begin{array}{ll}
\sqrt{n} & d=1 \\
\log(n) & d=2 \\
C_d & d\geq 3
\end{array}
\right.$$

Notice that for $d \geq 3$, $\left\langle T_n \right\rangle$ does **not** grow with $n$, which must mean that the walker somehow "escapes" and never returns back to the origin. The probability of return to the origin is less than 1! For an infinite-length random walk, indeed the probability of returning to the origin $\rho$ is seen to be

$$\rho \sim \left\{
\begin{array}{ll}
1 & d=1 \\
1 & d=2 \\
<1 & d\geq 3
\end{array}
\right.$$

The *intuitive* explanation of this amazing fact is that, as the dimension $d$ grows, there are "more directions available", and so more chances for the walker to "get lost" and never return to the origin. There is of course a formal proof as well, but today we will do a **computational verification** of these facts, which is no substitute for a formal proof but is often all we can do!
"""

# ╔═╡ e7f27328-b0ae-4c03-a6a2-b77f2c060f09
md"""
## Generating Random Walks
"""

# ╔═╡ 33dcd63e-a783-49eb-8e59-5c32f2d921b0
md"""
### Exercise 3.1
Write a function that generates a random walk of given length in $d$ dimensions. Your random walker should move as follows:

+ At each time-step, the walker moves only in one direction.
+ At each time-step, the walker moves only by -1 or +1
"""

# ╔═╡ 04034cd4-ab0f-4c60-bdb3-d83d1ab32205
function get_traj(length, dim)
   """Generate a RW in d dimensions

   Parameters
   ----------
   length: Int
       Length of the RW.
   dim: Int
       Dimension of the RW

   Returns
   -------
   traj : Matrix, (length, dim)
       The positions of the RW.

   Notes
   -----
   At each time-step, the walker moves in only one direction.
   At each time-step, the walker moves by -1 or +1
   """
   choices = hcat(Matrix(1.0I, dim, dim), Matrix(-1.0I, dim, dim))
   steps = vcat([@SVector(zeros(Float64, dim))], sample(svectors(choices, Val{dim}()), length-1))
   traj = cumsum(steps)
	return if dim > 1
   		reinterpret(reshape, Float64, traj) |> Transpose
	else
		reinterpret(reshape, Float64, traj)
	end
end;

# ╔═╡ 428a265a-f644-4fc4-a35f-060372e71de9
get_traj(; length=100, dim=2) = get_traj(length, dim);

# ╔═╡ 6dde4f1d-8ff5-46f1-b2d5-a9ab999fedc9
md"""
### Verification
To make sure that your function works correctly, execute the following cell. Notice the use of `assert` statements: execution should fail if something goes wrong. If everything is fine, nothing should happen.
"""

# ╔═╡ 9feeb518-dcff-4214-a77f-baee9a7cd348
md"""
### Exercise 3.2
Plot a random walk of length $10^4$ for $d=1$ (time in x-axis, position in y-axis) and $d=2$ (x,y components in x,y-axis). Remember to use **axis labels**.
"""

# ╔═╡ 50d91fb3-b77b-4bde-932b-5fb6e7f721e0
# it is better if you use one cell to generate the random walks, and a second cell to plot them
length = Int(1e4);

# ╔═╡ f205a339-6fe8-4588-9f97-7931f35b9242
md"""
## Counting the number of returns to the origin
Since we are interested in how **the expected number of returns to the origin** scales with the RW length, we don't need to store the whole trajectory of each simulation (we will be performing many simulations!). 
"""

# ╔═╡ d31cb06e-b4e7-4369-adb4-0635cbfb39f2
md"""
### Exercise 3.3
Write a function that generates a RW of given length and dimension (calling `get_traj`), and returns the number of times it returned to the origin. To count the number of returns to the origin, you might need to use the following functions:
```julia
all()
zeros()
```
"""

# ╔═╡ 96f8adc5-1bb3-406e-8107-e1644770b06d
md"""
### Exercise 3.4
Write a function that computes the expected number of returns to the origin for a given length and dimension. Your function will call `get_num_returns()`, and should have an additional parameter that sets the sample size.
"""

# ╔═╡ 67dedb1d-aa24-4c7f-8b3a-625156411570
get_average_num_returns(;length = 100, dim = 2, num_trajs=200) = get_average_num_returns(length, dim, num_trajs);

# ╔═╡ 99a18553-5d56-45e4-8d58-8d1b733d15b6
md"""
## Comparing with analytical results
We are now ready to compare our analytical results with numerical simulations! We want to plot the expected number of returns to the origin as a function of the RW length. To do this, it is useful to first define an array of RW lengths.
"""

# ╔═╡ af3e02f0-4c24-4250-b429-92008328e5d4
# define range of RW lengths
length_array = let length_min = 10, length_max = 100_000;
	# generate points logarithmically spaces
	# and convert them to integers
	[
		Int(round(x))
		for x in logrange(length_min, length_max, 20)
	]
end;

# ╔═╡ a1a218d2-9fb8-47b9-a9c3-62a739ae21e5
md"""
(tip: if your RW generating function is not very efficient, you might want to decrease `length_min`)  

Executing the following cell will run all simulations for $d=1$

"""

# ╔═╡ ab043acc-f824-4708-b7c0-d99dfd834b9d
md"""
### Exercise 3.5
Plot the average number of returns to the origin of a 1D RW as a function of the RW length, together with the expected theoretical result. Do your results verify the $n^{1/2}$ scaling? **Tip** Use double-logarithmic scales in your plot. Remember to include label axis, and a legend!
"""

# ╔═╡ a35aad44-9ab4-45fb-8a76-d88486f79667
md"""
### Exercise 3.6
Plot the average number of returns to the origin of a 2D RW as a function of the RW length. Do your results verify the $log(n)$ scaling? What are the best axis scales to use in this case?
"""

# ╔═╡ 1ece179a-ad15-4a19-8220-f58b6aee8a19
md"""
Notice that we had to put a multiplicative contant to the theoretical model to ease comparison with the numerical result
"""

# ╔═╡ 3c130766-a8bc-4b9e-83a5-72966dc38742
md"""
### Exercise 3.7
Show numerically that, for $d=3$ and $d=4$, the expected number of returns to the origin is **constant**.
"""

# ╔═╡ c3b63a33-72b2-493e-899e-024f6a3c706f
md"""
The two cells below will take approx. five minutes to execute both.
"""

# ╔═╡ f8496495-28a3-463e-9336-8c71bf3557fa
md"""
We see that, after an initial growth, the number of returns to origin stabilizes around a certain value.
"""

# ╔═╡ ff71bdc7-1314-4cb5-b456-174d661b3634
md"""
# Self-Avoiding Walks
Self-avoiding walks (SAW) are simply random walks in a regular lattice with the additional constraint that no point can be visited more than once. That is, SAWs cannot intersect themselves. The most well-known application of SAW is to model linear polymers, where obviously two monomers cannot occupy the same space (excluded volume effect).


You can read more about self-avoiding walks in this nice introduction by Gordon Slade:

[Self-Avoiding Walks, by Gordon Slade](https://www.math.ubc.ca/~slade/intelligencer.pdf)
"""

# ╔═╡ 4ac079b8-5b9b-4c12-99f3-07410da681d5
md"""
## Simulating Self-Avoiding Walks
Generating a SAW is not trivial. If you try to generate a SAW stochastically, that is, one step at a time, you will miserably fail: your walker might get into traps (configurations with no allowed movements), and if it does you will have to discard your simulation. It turns out you will have to discard your simulation *really* often, so that for large lengths, you will basically never find a valid path. In addition, the paths you will find for short lengths will not come up with the right probabilities. Bear in mind that we want to **uniformly sample** the set of SAW of given length $n$, SAW($n$). That is, we want that all paths from SAW($n$) are generated with the same probability.

The solution is to use a Monte Carlo algorithm that, given one element $\alpha \in \text{SAW}(n)$, generates a new one $\beta \in \text{SAW}(n)$ with some probability $P_{\alpha \beta}$. If in addition our algorithm satisfies **detailed balance** and is **ergodic**, then we known that it will converge to the equilibrium distribution (the uniform distribution in our case).

"""

# ╔═╡ 1b1fccdb-d511-4401-ba2e-b8610429cec4
md"""
## The pivot algorithm
We will implement the pivot algorithm, which is simple, effective, and satisfies detailed balance and ergodicity. You can read about the details of the pivot algorithm here:

[The Pivot Algorithm: A Highly Efficient Monte Carlo Method for the Self-Avoiding Walk](https://link.springer.com/article/10.1007/BF01022990)

(tip: if you're at home, **do not** use tools such as sci-hub to download the paper).

Given a self-avoiding walk of length $n$, the pivot algorithm generates the next walk $\beta \in \text{SAW}(n)$ as follows:

1. **Choose a point of $\alpha$ at random**, splitting the path in two bits: the head (from the origin to the chosen point) and the tail (from the chosen point to the end of the path). Notice that both the head and the tail are SAWs.
2. **Apply a transformation to the tail**, leaving the head intact. The transformation must be an orthoganl transformation that leaves the regular lattice intact (so, either a reflection or a $90º, 180º$ or $270º$ rotation). For simplicity, we will use only **rotations** (read the paper to see why this is ok).
3. **Check if the new path is self-avoiding**. If so, return it. Otherwise, return the original path.

Iterating these steps one obtains a **Markov** chain of SAWs: $\alpha_1 \to \alpha_2 \to \dots \to \alpha_M $. Notice that $\alpha_i$ are not uncorrelated, but because the algorithm satisfies detailed balance and is ergodic, we know that it approaches the equilibrium distribution. This means that we can use our Markov chain to compute **expected values** as long as it is long enough.
"""

# ╔═╡ 7dacf13b-de4f-4e3e-8a41-107dc9e31796
md"""
## Implementing the pivot step in 2D
To implement the **pivot algorithm** in 2D, we will write one function that does steps 1 and 2, and another function that does step 3. We will also need a function to generate standard 2D random walks.
"""

# ╔═╡ fa66b3cc-cf12-446f-b01a-ae58eab587ac
md"""
### Exercise 3.8
Write a function `get_traj` that generates a 2D random walk of given length.
"""

# ╔═╡ 455da7ca-4e6a-4678-a4e4-5e4bbe69599b
function get_traj(length)
	"""Generate a RW in 2 dimensions

	Parameters
	----------
	length: Int
		Length of the RW.

	Returns
	-------
	traj : Matrix, (length, 2)
		The positions of the RW.

	Notes
	-----
	At each time-step, the walker moves in only one direction.
	At each time-step, the walker moves by -1 or +1
	"""
	choices = hcat(Matrix(1.0I, 2, 2), Matrix(-1.0I, 2, 2))
	steps = vcat([@SVector(zeros(Float64, 2))], sample(svectors(choices, Val{2}()), length-1))
	traj = cumsum(steps)
	reinterpret(reshape, Float64, traj) |> Transpose
end;

# ╔═╡ 2b6d8254-238e-4ebd-8b26-ea12fb47f7fe
# basic checks for your RW generator
for dim in 1:5
    for length in [10, 100, 200, 500]
        traj = get_traj(length, dim)
		print(traj)
        # make sure traj has the right shape
		if dim > 1
        	@assert size(traj) == (length, dim)
		else
			@assert size(traj) == (length,)
		end
        #@assert all(isequal((dim,)) ∘ size, traj)
        # make sure all steps are -1 or 1 in only one direction
        @assert all(isone, sum(!iszero, diff(traj, dims=1), dims=2))
    end
end

# ╔═╡ 6167e084-3ddb-4fb0-8a67-252710ae0378
RW_1d = get_traj(length=length, dim=1);

# ╔═╡ c763bf62-6794-4a0b-b23a-0d56e43c49e7
RW_2d = get_traj(length=length, dim=2);

# ╔═╡ 5ff604e4-1bea-4360-9c7a-4b722eb20e98
begin
	# We generate a figure with two subplots, called axis in Plots.jl.
	
	p1 = plot(RW_1d, xlabel="time", ylabel="position", title="1d random walk")
	p2 = plot(RW_2d[:,1], RW_2d[:,2], xlabel = "x position", ylabel = "y position", title = "2d random walk")
	plot(p1, p2, layout=(1,2), legend=false)
end

# ╔═╡ 6b93e003-0e24-42c1-870b-9b195f589bd8
function get_num_returns(length, dim)
    # generate a RW of given length and dimension
    traj = get_traj(length, dim)
    # count how many times it goes through the origin
    count(iszero, eachrow(traj))
end;

# ╔═╡ b5553827-719e-4be2-ad81-7519d0d65076
function get_average_num_returns(length, dim, num_trajs=200)
    #average_num_returns
    return mean(get_num_returns(length, dim) for _ in 1:num_trajs)
end;

# ╔═╡ f1df514f-23d9-4977-bb41-3014c3e829b9
num_returns_array_1D = let dim=1, num_trajs = 200
	[
		get_average_num_returns(length=length, dim=dim, num_trajs=num_trajs)
		for length in length_array
	]
end;

# ╔═╡ cf0cdd18-650e-4a41-8937-4ab8f1ee1f00
let a = 0.8
	# plot theoretical result
	plot(length_array, [a * sqrt.(length_array), num_returns_array_1D], label=["Theory" "Numerics"], legend=true)
	# add axis labels
	xaxis!("traj length", :log10)
	yaxis!("# returns to origin", :log10)
	# add a title (e.g. that says what dimension we used)
	title!("Returns to origin vs. length - 1D walk")
end

# ╔═╡ 960674a7-d0d6-401f-b128-a721b3fbfc91
num_returns_array_2D = let dim = 2, num_trajs = 200
	[
		get_average_num_returns(length=length, dim=dim, num_trajs=num_trajs)
		for length in length_array
	]
end;

# ╔═╡ 9e109fc1-a849-4622-b1d6-38461ee5847e
# plot theoretical result
let k = 0.3, h = 1
	plot(length_array, [h .+ k * log.(length_array), num_returns_array_2D], label=["Theory" "Numerics"], legend=true)
	# add axis labels
	xaxis!("traj length", :log10)
	yaxis!("# returns to origin")
	# add a title (e.g. that says what dimension we used)
	title!("Returns to origin vs. length - 2D walk")
end

# ╔═╡ f43a2467-2101-4b6d-aa91-4030681dff07
# do the simulations for d=3
num_returns_array_3D = let dim = 3, num_trajs = 20_000
	[
		get_average_num_returns(length, dim, num_trajs)
		for length in length_array
	]
end;

# ╔═╡ 2966a511-92d3-4f3c-ae70-ac8d82a25e60
# do the simulations for d=4
num_returns_array_4D = let dim = 4, num_trajs = 20_000
	[
		get_average_num_returns(length, dim, num_trajs)
		for length in length_array
	]
end;

# ╔═╡ 37cdef7f-a5dd-4a90-a534-a1526a3238f8
begin
	# plot theoretical result
	plot(length_array, [num_returns_array_3D, num_returns_array_4D], label=["Numeric 3D" "Numeric 4D"], legend=:topleft)
	# add axis labels
	xaxis!("traj length", :log10)
	yaxis!("# returns to origin")
	# add a title (e.g. that says what dimension we used)
	title!("Returns to origin vs. length - 3D and 4D walk")
end

# ╔═╡ d1c3baba-f52f-465c-9414-e18bbda12769
function rotation_matrix(direction, point=[0,0])
    """Return a 2D matrix that rotates 90°, 180° or 270°"""
    @assert direction in (0, 1, 2)
	@assert size(point) == (2,)
	(x,y) = point
	(sinθ, cosθ) = if direction == 0
		(1, 0)
	elseif direction == 1
		(0, -1)
	elseif direction == 2
		(-1, 0)
	end
    [ cosθ -sinθ (-x*cosθ + y*sinθ + x)
	  sinθ  cosθ (-x*sinθ - y*cosθ + y)
	   0     0              1          ]
end;

# ╔═╡ ebc7426b-c6af-4d06-b539-8603a942ad11
function pivot_traj(traj)
	@assert size(traj, 1) > 1 "Trajectory must be at least 2 steps long"
    pivot = rand(1:size(traj, 1)-1)
	(head, tail) = (traj[begin:pivot,:], traj[pivot+1:end,:])
	tail = (hcat(tail, ones(size(tail, 1))) * transpose(rotation_matrix(rand(0:2), head[end,:])))[:, 1:2]
    return vcat(head, tail)
end;

# ╔═╡ 0e8f6107-e195-41e6-9932-51d7a4621c6d
md"""
### Exercise 3.10
Write a function that counts the number of self-intersections of a RW. Notice that SAWs have 0 self intersections, so that will solve step 3 of the pivot algorithm, but will also be useful to generate the initial condition. One way of approaching this exercise is to count how many *different* points the path visits.
"""

# ╔═╡ 22aab5b0-c7f1-4978-bba2-bf23942e0145
function count_self_intersections(traj)
    """Count the number of self-intersections of a RW"""
	size(traj, 1) - size(unique(traj, dims=1), 1)
end;

# ╔═╡ 99700da2-44ae-475f-bfb2-5b04646fa31c
md"""
I also write a function to quickly check if there are any self-intersections
"""

# ╔═╡ 085b6630-7c4d-4248-9a30-9e4cff15d6b8
function isSAW(traj)
	"""Check if a RW has any self-intersections"""
	allunique(eachrow(traj))
end;

# ╔═╡ 6f0df9ff-d0f2-46fc-99ba-fe9b1cada41d
md"""
### Exercise 3.11
Verify that your `count_self_intersecitons` function works properly by using short trajectories for which you know the answer.
"""

# ╔═╡ 32f97af6-2f06-4d8f-8af3-0f906a8ec5e4
traj = [0. 0.
        0. 1.
        1. 1. #
        1. 2. #
        2. 2. #
        1. 2. # 1st intersection
        1. 1. # 2nd intersection
        2. 1.
        2. 2. # 3rd intersection
        3. 2. #
        3. 1.
        3. 2. # 4th intersection
        3. 3. #
        2. 3. #
        1. 3. #
        1. 2. # 5th intersection
        1. 3. # 6th intersection
        2. 3. # 7th intersection
        3. 3. # 8th intersection
        3. 2. # 9th intersection
        4. 2.];

# ╔═╡ 15d03596-d033-4bb5-96cc-51a316a4e314
md"""
We can count 9 intersections in the trajectory above
"""

# ╔═╡ a4ead578-ae90-45c0-b6ca-76c4b61a3fcb
count_self_intersections(traj)

# ╔═╡ e0e7e7a6-fb30-4f5f-a036-14016cc59006
md"""
The verification is thus complete.
"""

# ╔═╡ 9a70865c-7183-4651-9dd5-6c2528bbfa4c
md"""
## Generating the initial condition
You might have noticed that the pivot algorithm requires an element of SAW($n$) as starting condition, to then generate a Markov chain easily. But how do you get this first element? We will use the following strategy:
1. Generate a standard 2D random walk, and count the number of self intersections.
2. Apply the pivot transformation to get a new 2D random walk, and count the number of self-intersections-
3. If the number of self-intersections has decreased or not changed, keep the new path. Else, keep the old one.
4. Go to 2, till the number of self-intersections is 0.

### Exercise 3.12
Write a function `get_first_SAW` that generates a SAW of given length
"""

# ╔═╡ 8a15df2f-4ec9-4866-9dca-0d31a0785b51
function get_first_SAW(length, max_tries=1_000_000)
    traj = get_traj(length)
    traj_inter = count_self_intersections(traj)
    for _ in 1:max_tries
        if traj_inter == 0
            return traj
		end
        candidate = pivot_traj(traj)
        candidate_inter = count_self_intersections(candidate)
        if candidate_inter ≤ traj_inter
            traj = candidate
            # if candidate_inter < traj_inter: # This is just to have an indicator of progress
            #    print(candidate_inter)
            traj_inter = candidate_inter
		end
	end
    error("No SAW found after $max_tries tries")
end;

# ╔═╡ c84d1b9f-e438-46a9-adcf-7d2bd840c31b
md"""
### Exercise 3.13
Generate and plot some 2D SAWs of different lengths. Be carefull, raise the length slowly! You can measure how long a cell takes executing watching the time indicator at the bottom-right of a cell.
"""

# ╔═╡ f4aa1d61-eb44-4ec1-952d-58d62ec8817a
md"""
CAUTION!!! Some of the generated RW may have configurations such that it is unlikely that they will be pivoted into a SAW within `max_tries` attempts. In such cases trying again may lead to success, in particular with shorter walks.
"""

# ╔═╡ 446263fa-c7cc-4e03-8f14-160e4e96fd09
@timeit	SAW05 = get_first_SAW(5)

# ╔═╡ 32df7409-575f-4dab-b307-9c044eaf40b8
plot(SAW05[:, 1], SAW05[:,2], xlabel = "x", ylabel = "y", title = "SAW 05", legend=false)

# ╔═╡ 903e3657-6939-4666-8f4e-600807973865
@timeit	SAW10 = get_first_SAW(10)

# ╔═╡ 70d4e564-8c57-4413-b1f4-8b38cd530c66
plot(SAW10[:, 1], SAW10[:,2], xlabel = "x", ylabel = "y", title = "SAW 10", legend=false)

# ╔═╡ a327a8ad-4d83-465d-89f3-1e5c15d29899
@timeit	SAW100 = get_first_SAW(100)

# ╔═╡ ceec461a-3608-4bd0-ae1f-082bcbdb1e93
plot(SAW100[:, 1], SAW100[:,2], xlabel = "x", ylabel = "y", title = "SAW 100", legend=false)

# ╔═╡ 2797e4d0-56e7-4292-bb44-5cc84d439ea4
@timeit SAW500 = get_first_SAW(500)

# ╔═╡ e3a425df-ccbc-4c85-ad43-5532fa1a47a0
plot(SAW500[:, 1], SAW500[:,2], xlabel = "x", ylabel = "y", title = "SAW 500", legend=false)

# ╔═╡ fd9cc739-6413-4360-a9c0-2f97fcfe401a
@timeit SAW1_000 = get_first_SAW(1_000)

# ╔═╡ 3de64238-6535-405c-b431-3dddd1e1abc2
plot(SAW1_000[:, 1], SAW1_000[:,2], xlabel = "x", ylabel = "y", title = "SAW 1_000", legend=false)

# ╔═╡ 5b0c024d-e1f4-4573-9fe3-bee937eddb1f
@timeit SAW5_000 = get_first_SAW(5_000)

# ╔═╡ 7f972cc2-2f14-4aed-8f18-217982655c48
plot(SAW5_000[:, 1], SAW5_000[:,2], xlabel = "x", ylabel = "y", title = "SAW 5_000", legend=false)

# ╔═╡ f33bd10e-9594-4e1c-acab-91de76f21bbd
md"""
### Exercise 3.14
Write a function `get_next_SAW` that, given a SAW, generates another SAW using the pivot algorithm. Your function should check that the input RW is really a SAW. Remember the steps:

1. Apply the pivot transformation
2. Check if the new path is self-avoiding. **If so, return it. Otherwise, return the original path.**
"""

# ╔═╡ 569757aa-d6d3-41ae-b28f-8101cb2f4c78
function get_next_SAW(traj)
    # make sure input traj is SAW
    @assert isSAW(traj)
    
    # pivot step
    proposed_traj = pivot_traj(traj)
    
    # if it's a SAW
    isSAW(proposed_traj) ? 
		proposed_traj : 
		traj
end;

# ╔═╡ 2ee69605-63ef-46b3-99f2-57796c53d4c9
md"""
### Mean Squared Displacement in SAW
A quantity of interest in RWs is the mean squared displacement, which is simply the (squared) distance between the endpoints of the walk. Usually, one writes

$$\left\langle X(n)^2 \right\rangle \sim n^{2 \nu}$$

As you know, for a standard RW of $n$ steps, the mean-squared displacement scales like $n$, so $\nu=1/2$. However, the exponent for SAW is **different**! Althought it has not been formally proven (still), it is believed that the exponent for SAW is $\nu=3/4$. That is, for a self-avoiding random walk, the mean squared displacement scales as $n^{3/2}$.
"""

# ╔═╡ fba46220-3c9c-4401-895a-a0aa59517494
md"""
### Exercise 3.15
Explain why it makes sense that the mean-squared displacement exponent of SAW is **greater** than that of standard RW. 
"""

# ╔═╡ a155b069-8abf-4582-962f-db29ff057e0f
md"""
#### Explanation
The thesis is sensible since we expect the self-avoidance constraint to act as a repulsor from the points already visited, thus increasing displacement proportionally to the number of already visited points.

A more quantitative approach might be to consider the tightest (i.e. least displaced) and loosest (i.e. most displaced) walks and see what they tell us about the lower and upper limits of the scale relation between displacement and walk length. 

Let us assume that exists a SAW that completely fills all $n$ lattice points enclosed within a certain distance $R$ from the origin: for sufficiently high $n$ the number of lattice points in $n \propto R^2$, therefore $R \propto n^{1/2}$. Since we also know that, for an isotropic distribution in a finite space of radius $R$ the mean-squared displacement is $\left\langle X(n)^2 \right\rangle \propto R^2$, then:

$$\left\langle X(n)^2 \right\rangle \propto n$$

which implies $\nu > \frac{1}{2}$, since space-filling walks are a lower bound case for displacement.

Then we consider the most displaced SAW: a straight walk. It is easy to calculate that in this case

$$\left\langle X(n)^2 \right\rangle \propto n^2$$

which implies that $\nu < 1$ since straight walks are an upper bound case for displacement.

Therefore we have demonstrated that 

$$\nu \in \left(1/2,1\right)$$

"""

# ╔═╡ ef522cf2-76a4-4ae6-b530-1ab373ccb212
md"""
### Exercise 3.16
Verify numerically the scaling of the mean-squared displacement of SAW. Notice that you don't need to store all the SAWs, just the endpoints. You could follow this scheme:

1. Generate a first SAW with your `get_first_SAW()` function
2. Generate the next SAW using your `get_next_SAW()` function, and store the endpoint.
3. Iterate step 2 for as many steps as required
4. Compute the average mean-squared displacement of the stored endpoints

Then repeating steps 1-4 for different lengths, and plot the results in double-logarithmic axis. Compare your results with the theoretical exponent. Do they agree?
"""

# ╔═╡ 77e233ad-7cc4-4261-8a73-ce392d61781c
function get_average_displacement(N, steps=1000)
	SAW = nothing
	sq_displacements = map(1:steps) do _
		SAW = if isnothing(SAW)
			get_first_SAW(N)
		else
		    get_next_SAW(SAW)
		end
		norm(SAW[end, :])^2
	end
    return mean(sq_displacements)
end;

# ╔═╡ ed894edf-7a79-4b43-ab8d-8cf950eed057
walk_length_array = let length_min = 5, length_max = 500
	[
		Int(round(x))
		for x in logrange(length_min, length_max, 20)
	]
end;

# ╔═╡ f8c3f04f-40e9-42ca-82bd-c4226899451d
average_displacement_array = [get_average_displacement(length) for length in walk_length_array];

# ╔═╡ 4d8a5f53-c3a3-4dd7-a583-b5ceec021f57
let a = 0.75
	plot(walk_length_array, [a * walk_length_array .^(3//2), average_displacement_array], label=["Theory" "Numerics"], legend=true)
	xaxis!("N", :log10)
	yaxis!("<X²>", :log10)
	title!("Mean-squared Displacement vs. Walk Lenght")
end

# ╔═╡ 6a6c5826-cfc9-4d2b-a41f-ee72b1fe58bf
md"""
As can be seen in the plot above, the theoretically predicted power law is consistent with observed behavior.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
Plots = "~1.23.5"
PlutoUI = "~0.7.18"
StaticArrays = "~1.2.13"
StatsBase = "~0.33.12"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "0ec322186e078db08ea3e7da5b8b2885c099b393"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.0"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f885e7e7c124f8c92650d61b9477b9ac2ee607dd"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.1"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dce3e3fea680869eaa0b774b2e8343e9ff442313"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.40.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "30f2b340c2fff8410d89bfcdc9c0a6dd661ac5f7"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.62.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fd75fa3a2080109a2c0ec9864a6e14c60cca3866"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.62.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "14eece7a3308b4d8be910e265c724a6ba51a9798"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.16"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "f0c6489b12d28fb4c2103073ec7452f3423bd308"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.1"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "6193c3815f13ba1b78a51ce391db8be016ae9214"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.4"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun"]
git-tree-sha1 = "7dc03c2b145168f5854085a16d054429d612b637"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.23.5"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "57312c7ecad39566319ccf5aa717a20788eb8c1f"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.18"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "eb35dcc66558b2dda84079b9a1be17557d32091a"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.12"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╠═2848a983-6b48-4d81-b451-d3290f931426
# ╠═8b2869aa-9315-4aac-a2a4-18c33be82cae
# ╠═16d76581-01d3-40ab-a72a-eb2543ba9859
# ╠═1fa3fa02-50fb-40d1-b7fa-9ae52efa3d16
# ╟─5d3095fe-72d2-43ef-9a63-7f67518b85eb
# ╟─e7f27328-b0ae-4c03-a6a2-b77f2c060f09
# ╟─33dcd63e-a783-49eb-8e59-5c32f2d921b0
# ╠═04034cd4-ab0f-4c60-bdb3-d83d1ab32205
# ╠═428a265a-f644-4fc4-a35f-060372e71de9
# ╟─6dde4f1d-8ff5-46f1-b2d5-a9ab999fedc9
# ╠═2b6d8254-238e-4ebd-8b26-ea12fb47f7fe
# ╟─9feeb518-dcff-4214-a77f-baee9a7cd348
# ╠═50d91fb3-b77b-4bde-932b-5fb6e7f721e0
# ╠═6167e084-3ddb-4fb0-8a67-252710ae0378
# ╠═c763bf62-6794-4a0b-b23a-0d56e43c49e7
# ╠═5ff604e4-1bea-4360-9c7a-4b722eb20e98
# ╟─f205a339-6fe8-4588-9f97-7931f35b9242
# ╟─d31cb06e-b4e7-4369-adb4-0635cbfb39f2
# ╠═6b93e003-0e24-42c1-870b-9b195f589bd8
# ╟─96f8adc5-1bb3-406e-8107-e1644770b06d
# ╠═b5553827-719e-4be2-ad81-7519d0d65076
# ╠═67dedb1d-aa24-4c7f-8b3a-625156411570
# ╟─99a18553-5d56-45e4-8d58-8d1b733d15b6
# ╠═af3e02f0-4c24-4250-b429-92008328e5d4
# ╟─a1a218d2-9fb8-47b9-a9c3-62a739ae21e5
# ╠═f1df514f-23d9-4977-bb41-3014c3e829b9
# ╟─ab043acc-f824-4708-b7c0-d99dfd834b9d
# ╠═cf0cdd18-650e-4a41-8937-4ab8f1ee1f00
# ╟─a35aad44-9ab4-45fb-8a76-d88486f79667
# ╠═960674a7-d0d6-401f-b128-a721b3fbfc91
# ╠═9e109fc1-a849-4622-b1d6-38461ee5847e
# ╟─1ece179a-ad15-4a19-8220-f58b6aee8a19
# ╟─3c130766-a8bc-4b9e-83a5-72966dc38742
# ╟─c3b63a33-72b2-493e-899e-024f6a3c706f
# ╠═f43a2467-2101-4b6d-aa91-4030681dff07
# ╠═2966a511-92d3-4f3c-ae70-ac8d82a25e60
# ╠═37cdef7f-a5dd-4a90-a534-a1526a3238f8
# ╟─f8496495-28a3-463e-9336-8c71bf3557fa
# ╟─ff71bdc7-1314-4cb5-b456-174d661b3634
# ╟─4ac079b8-5b9b-4c12-99f3-07410da681d5
# ╟─1b1fccdb-d511-4401-ba2e-b8610429cec4
# ╟─7dacf13b-de4f-4e3e-8a41-107dc9e31796
# ╟─fa66b3cc-cf12-446f-b01a-ae58eab587ac
# ╠═455da7ca-4e6a-4678-a4e4-5e4bbe69599b
# ╠═d1c3baba-f52f-465c-9414-e18bbda12769
# ╠═ebc7426b-c6af-4d06-b539-8603a942ad11
# ╟─0e8f6107-e195-41e6-9932-51d7a4621c6d
# ╠═22aab5b0-c7f1-4978-bba2-bf23942e0145
# ╟─99700da2-44ae-475f-bfb2-5b04646fa31c
# ╠═085b6630-7c4d-4248-9a30-9e4cff15d6b8
# ╟─6f0df9ff-d0f2-46fc-99ba-fe9b1cada41d
# ╠═32f97af6-2f06-4d8f-8af3-0f906a8ec5e4
# ╟─15d03596-d033-4bb5-96cc-51a316a4e314
# ╠═a4ead578-ae90-45c0-b6ca-76c4b61a3fcb
# ╟─e0e7e7a6-fb30-4f5f-a036-14016cc59006
# ╟─9a70865c-7183-4651-9dd5-6c2528bbfa4c
# ╠═8a15df2f-4ec9-4866-9dca-0d31a0785b51
# ╟─c84d1b9f-e438-46a9-adcf-7d2bd840c31b
# ╟─f4aa1d61-eb44-4ec1-952d-58d62ec8817a
# ╠═446263fa-c7cc-4e03-8f14-160e4e96fd09
# ╠═32df7409-575f-4dab-b307-9c044eaf40b8
# ╠═903e3657-6939-4666-8f4e-600807973865
# ╠═70d4e564-8c57-4413-b1f4-8b38cd530c66
# ╠═a327a8ad-4d83-465d-89f3-1e5c15d29899
# ╠═ceec461a-3608-4bd0-ae1f-082bcbdb1e93
# ╠═2797e4d0-56e7-4292-bb44-5cc84d439ea4
# ╠═e3a425df-ccbc-4c85-ad43-5532fa1a47a0
# ╠═fd9cc739-6413-4360-a9c0-2f97fcfe401a
# ╠═3de64238-6535-405c-b431-3dddd1e1abc2
# ╠═5b0c024d-e1f4-4573-9fe3-bee937eddb1f
# ╠═7f972cc2-2f14-4aed-8f18-217982655c48
# ╟─f33bd10e-9594-4e1c-acab-91de76f21bbd
# ╠═569757aa-d6d3-41ae-b28f-8101cb2f4c78
# ╟─2ee69605-63ef-46b3-99f2-57796c53d4c9
# ╟─fba46220-3c9c-4401-895a-a0aa59517494
# ╟─a155b069-8abf-4582-962f-db29ff057e0f
# ╟─ef522cf2-76a4-4ae6-b530-1ab373ccb212
# ╠═77e233ad-7cc4-4261-8a73-ce392d61781c
# ╠═ed894edf-7a79-4b43-ab8d-8cf950eed057
# ╠═f8c3f04f-40e9-42ca-82bd-c4226899451d
# ╠═4d8a5f53-c3a3-4dd7-a583-b5ceec021f57
# ╟─6a6c5826-cfc9-4d2b-a41f-ee72b1fe58bf
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
