import numpy as np

from random import uniform

# set numpy error
np.seterr( over='raise' )

# read full file
data = open( 'pg_essays.txt', 'rt' ).read()

# dictionary conversion. We will use n-gram coding for all the unique characters in the input data
dict_chars = list( set( data ) )
data_size = len( data )
dict_size = len( dict_chars )

# character encoding
char_to_x = { c:i for i, c in enumerate( dict_chars ) }
x_to_char = { i:c for i, c in enumerate( dict_chars ) }

print( 'data size: {0}, dictionary size: {1}'.format( data_size, dict_size ) )

# net structure
hidden_nodes = 200
seq_len = 25

dropout_prob = 0.5

learning_rate = 1e-1

# model weights - drawn initially from guassian distribution
Whh = np.random.normal( 0., 0.01, (hidden_nodes, hidden_nodes) )
Wxh = np.random.normal( 0., 0.01, (hidden_nodes, dict_size ) )
Why = np.random.normal( 0., 0.01, (dict_size, hidden_nodes ) )

bh = np.zeros( (hidden_nodes, 1) )
by = np.zeros( (dict_size, 1 ) )

# calculate loss, model weights gradients and return hidden state for propagation
def rnn( inputs, targets, hprev ):
    '''
    input and target have len of seq_len
    '''
    
    xs, hraw, hs, ys, ps = {}, {}, {}, {}, {}
    hs[ -1 ] = hprev
    loss = 0.

    dropout_table = np.random.binomial( 1, dropout_prob, (seq_len, hidden_nodes, 1) )
    
    # forward pass
    for i in range( len( inputs ) ):

		# encode input character
		xs[i] = np.zeros( ( dict_size, 1 ) )
		xs[i][inputs[i]] = 1.
		
		hraw[i] = np.dot( Wxh, xs[i] ) + np.dot( Whh, hs[i-1] ) + bh
		hraw[i] = hraw[i] * dropout_table[i]

		hs[i] = np.maximum( hraw[i], 0. )
		ys[i] = np.dot( Why, hs[i] ) + by
		
		# clip ys to avoid overflows.  tanh does clipping via it's natural range
		np.clip( ys[i], -100., 100., out=ys[i] )
		
		# normalise probabilities
		ps[i] = np.exp( ys[i] ) / np.sum( np.exp( ys[i] ) )
		
		# softmax (cross-entropy loss)
		loss += -np.log( ps[i][targets[i],0] )

    dWxh, dWhh, dWhy = np.zeros_like( Wxh ), np.zeros_like( Whh ), np.zeros_like( Why )
    dbh, dby = np.zeros_like( bh ), np.zeros_like( by )
    dhnext = np.zeros_like( hs[0] )
    
    # backward pass: start from the end
    for i in reversed( range( len( inputs ) ) ):
        dy = np.copy( ps[i] )
        dy[targets[i]] -= 1.0    # backprop into y.  In the target state the ps[ target[i] ] == 1.0

        # recover weights
        dWhy += np.dot( dy, hs[i].T )
        dby += dy
        
        dh = np.dot( Why.T, dy) + dhnext # backprop into h
        dhtemp = np.zeros_like( dhnext )
        dhtemp[ hraw[i] > 0. ] = 1.
        dhraw = dhtemp * dh
        
        dbh += dhraw
        dWxh += np.dot( dhraw, xs[i].T )
        dWhh += np.dot( dhraw, hs[i-1].T )
        dhnext = np.dot( Whh.T, dhraw )
        
        # clip to mitigate exploding gradients
        for dparam in [ dWxh, dWhh, dWhy, dbh, dby ]:
            np.clip( dparam, -5, 5, out=dparam )
            
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[ len(inputs)-1 ]


# gradient validation
def gradCheck(inputs, targets, hprev):
    global Wxh, Whh, Why, bh, by

    num_checks, delta = 10, 1e-5

    # calculate gradients using backprop
    _, dWxh, dWhh, dWhy, dbh, dby, _ = rnn( inputs, targets, hprev )

    for param, dparam, name in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby], ['Wxh', 'Whh', 'Why', 'bh', 'by']):
        s0 = dparam.shape
        s1 = param.shape
        assert s0 == s1, 'Error dims dont match: %s and %s.' % (`s0`, `s1`)
        print( name )

        for i in xrange(num_checks):
            ri = int( uniform(0,param.size) )
            
            # evaluate cost at [x + delta] and [x - delta]
            old_val = param.flat[ri]
            param.flat[ri] = old_val + delta
            cg0, _, _, _, _, _, _ = rnn( inputs, targets, hprev )
            param.flat[ri] = old_val - delta
            cg1, _, _, _, _, _, _ = rnn( inputs, targets, hprev )
            param.flat[ri] = old_val # reset old value for this parameter
 
            # fetch both numerical and analytic gradient
            grad_analytic = dparam.flat[ri]
            grad_numerical = (cg0 - cg1) / ( 2 * delta )
            if grad_numerical + grad_analytic == 0.0:
                rel_error = 0.0
            else:
                rel_error = abs( grad_analytic - grad_numerical ) / abs( grad_numerical + grad_analytic )

            print( '%f, %f => %e ' % ( grad_numerical, grad_analytic, rel_error) )
            # rel_error should be on order of 1e-7 or less

def sample( h, seed_ix, n ):
	""" 
	sample a sequence of integers from the model 
	h is memory state, seed_ix is seed letter for first time step
	"""
	x = np.zeros( (dict_size, 1) )
	x[ seed_ix ] = 1
	ixes = []
	for t in range( n ):
		h = np.maximum( np.dot( Wxh, x ) + np.dot( Whh, h ) + bh, 0. )
		y = np.dot( Why, h ) + by
		np.clip( y, -100., 100., out=y )
		p = np.exp( y ) / np.sum( np.exp( y ) )
		ix = np.random.choice( range( dict_size ), p=p.ravel() )
		x = np.zeros( (dict_size, 1) )
		x[ix] = 1
		ixes.append(ix)

	return ixes

# run gradient validation
if True:
	p = 0
	inputs = [ char_to_x[ c ] for c in data[ p:p+seq_len ] ]
	targets = [ char_to_x[ c ] for c in data[ p + 1:p+seq_len + 1 ] ]

	hprev = np.zeros_like( bh )

	print( data[ p:p+seq_len ], inputs )
	print( data[ p + 1:p+seq_len + 1 ], targets )

	gradCheck( inputs, targets, hprev )

	import sys
	sys.exit( 0 )

# main program
n, p = 0, 0

# memory for Adagrad
mWxh = np.zeros_like( Wxh )
mWhh = np.zeros_like( Whh )
mWhy = np.zeros_like( Why )

mbh, mby = np.zeros_like(bh), np.zeros_like(by)

# loss at iteration 0
smooth_loss = -np.log(1.0/dict_size)*seq_len

while True:
	# prepare inputs (we're sweeping from left to right in steps seq_length long)
	if p+seq_len+1 >= len(data) or n == 0:
		hprev = np.zeros( (hidden_nodes,1) ) # reset RNN memory
		p = 0 # go from start of data

	inputs = [char_to_x[ch] for ch in data[p:p+seq_len]]
	targets = [char_to_x[ch] for ch in data[p+1:p+seq_len + 1]]

	# sample from the model now and then
	if n % 100 == 0:
		sample_ix = sample(hprev, inputs[0], 200)
		txt = ''.join(x_to_char[ix] for ix in sample_ix)
		print( '----\n %s \n----' % txt )

	# forward seq_length characters through the net and fetch gradient
	loss, dWxh, dWhh, dWhy, dbh, dby, hprev = rnn( inputs, targets, hprev )
	smooth_loss = smooth_loss * 0.999 + loss * 0.001
	if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress

	# perform parameter update with Adagrad
	for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
								[dWxh, dWhh, dWhy, dbh, dby], 
								[mWxh, mWhh, mWhy, mbh, mby]):
		mem += dparam * dparam
		param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

	p += seq_len # move data pointer
	n += 1 # iteration counter 
