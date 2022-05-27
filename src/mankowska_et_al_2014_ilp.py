import os
import copy
import numpy as np
import scipy.io
from datetime import datetime
from pulp import *


def solve_mk_model(nbNodes, nbVehi, nbServi, r, DS, a, x, y, d, p, mind, maxd, e, l):
	'''
	Implement model from Mankowska et al., based
	on their own instance data
	'''

	### STEP 1:
	#   Arrange data in dicts
	print('Preparing data...')

	TOLERANCE = 1e-4
	bigM = 1e5
	fixedP = p[1][1][1] # In practice, p is the same for all jobs, so avoid dictionary and use this
	gurobi_gap = 0.001
	gurobi_time_limit = 3600*8

	# Define the sets for skills, jobs and nurses
	S = ['s_' + str(i) for i in range(nbServi)]
	V = ['n_' + str(i) for i in range(nbVehi)]
	C = ['j_' + str(i + 1) for i in range(nbNodes - 2)]
	C_0 = ['j_' + str(i) for i in range(nbNodes - 1)]
	C_d = ['j_' + str(int(i - 1)) for i in DS if i > 0]

	depot = C_0[0]
	# return_depot = C_0[-1]

	# Create dictionaries from the data:
	a_dict = makeDict([V, S], a, 0)
	r_dict = makeDict([C_0, S], r, 0)
	d_dict = makeDict([C_0, C_0], d, 0)
	mind_dict = makeDict([C_0], mind, 0)
	maxd_dict = makeDict([C_0], maxd, 0)
	e_dict = makeDict([C_0], e, 0)
	l_dict = makeDict([C_0], l, 0)

	### STEP 2:
	#   Create problem and variables
	print('Creating variables...')

	prob = LpProblem("Mankowska_Model_" + str(nbNodes) + '_' + str(nbVehi) + '_' + str(nbServi), LpMinimize)

	# X - If an arc is used or not
	x_domain = (C_0, C_0, V, S)
	x = LpVariable.dicts("x", x_domain, 0,None,LpInteger) 

	# T - time when a job starts
	t_domain = (C_0, V, S)
	t = LpVariable.dicts("t", t_domain,lowBound=0,cat='Continuous') 

	# Z - tardiness of a job
	z_domain = (C_0, S)
	z = LpVariable.dicts("z", z_domain, lowBound=0, cat='Continuous') 

	# Distance travelled
	D = LpVariable("D", 0)
	T = LpVariable("T", 0)
	Tmax = LpVariable("Tmax", 0)


	### STEP 3:
	#   Add constraints
	print('Adding constraints...')

	# Constraints (2) in paper
	prob += D == lpSum(d_dict[i][j]*x[i][j][v][s] for i in C_0 for j in C_0 for v in V for s in S),"Variable D"

	# Constraints (3) in paper
	prob += T == lpSum(z[i][s] for i in C_0 for s in S),"Variable T"

	# Constraints (4) in paper
	for i in C:
		for s in S:
			prob += Tmax >= z[i][s],"Variable Tmax job_" + str(i) + '_skill_' + str(s)


	# Constraints (5) in paper
	for v in V:
		prob += lpSum([x[depot][i][v][s] for i in C_0 for s in S]) == 1, "Start from depot nurse " + str(v)
		prob += lpSum([x[i][depot][v][s] for i in C_0 for s in S]) == 1, "Finish at depot nurse " + str(v)

	# Constraints (6) in paper
	ct_num = 0
	for i in C:
		for v in V:
			ct_num += 1
			prob += lpSum([x[j][i][v][s] for j in C_0 for s in S]) == lpSum([x[i][j][v][s] for j in C_0 for s in S]), "Inflow_Outflow_" + str(ct_num) + "_job_" + str(i) + '_nurse_' + str(v)

	# Constraints (7) in paper
	ct_num = 0
	for i in C:
		for s in S:
			ct_num += 1
			prob += lpSum([x[j][i][v][s]*a_dict[v][s] for j in C_0 for v in V]) == r_dict[i][s], "Assign to qualified caregiver" + str(ct_num) + "_job_" + str(i) + '_skill_' + str(s)

	p_value = 0
	# Constraints (8) in paper
	for i in C_0:
		if i == 'j_0':
			p_value = 0
		else:
			p_value = fixedP
		for j in C:
			for v in V:
				for s1 in S:
					for s2 in S:
						prob += t[i][v][s1] + p_value + d_dict[i][j] <= t[j][v][s2] + bigM*(1 - x[i][j][v][s2])

	# Prevent depot to depot trips
	# for v in V:
	# 	prob += lpSum([x[depot][depot][v][s] for s in S]) == 0, "Cycle on depot forbidden N" + str(v)
	# 	prob += lpSum([x[return_depot][return_depot][v][s] for s in S]) == 0, "Cycle on return depot forbidden N" + str(v)


	# Constraints (9) and (10) in paper
	for i in C_0:
		for v in V:
			for s in S:
				prob += t[i][v][s] >= e_dict[i]
				prob += t[i][v][s] <= l_dict[i] + z[i][s]

	# Constraints (11) and (12) in paper
	for i in C_d:
		for v1 in V:
			for v2 in V:
				for si1, s1 in enumerate(S):
					for si2 in range(si1 + 1, len(S)):
						s2 = S[si2]
						ctname = str(i) + "_" + str(v1) + "_" + str(v2) + "_" + str(s1) + "_" + str(s2)
						prob += t[i][v2][s2] - t[i][v1][s1] >= mind_dict[i] - bigM*(2 - lpSum([x[j][i][v1][s1] for j in C_0]) - lpSum([x[j][i][v2][s2] for j in C_0])), "Mind_" + ctname
						prob += t[i][v2][s2] - t[i][v1][s1] <= maxd_dict[i] + bigM*(2 - lpSum([x[j][i][v1][s1] for j in C_0]) - lpSum([x[j][i][v2][s2] for j in C_0])), "Maxd_" + ctname

	# Constraints (13) in paper
	for i in C_0:
		for j in C_0:
			for v in V:
				for s in S:
					prob += x[i][j][v][s] <= a_dict[v][s]*r_dict[j][s], "Restrict domain X " + str(i) + "_"+ str(j) + "_" + str(v) + "_" + str(s)


	# Constraints (14) in paper
	for i in C_0:
			for s in S:
				if (r_dict[i][s] < 0.5):
					prob += z[i][s] == 0, "Reduce domain z" + str(i) + str(v) + str(s)
				else:
					prob += z[i][s] >= 0

	### STEP 4:
	#   Add objective function
	print('Adding objective function...')
	# Objective
	prob += 1/3*(D + T + Tmax),"Objective"


	#Debug obj:
	# prob += 1/3*(D + T + Tmax) -0.000001*lpSum([t[i][v][s] for i in C_0 for v in V for s in S]),"minimise Ts"
	

	### STEP 5:
	#   SOLVE
	print('Solving...')
	# prob.writeLP("model.lp")

	# prob.solve()
	prob.solve(GUROBI(epgap = gurobi_gap, timeLimit=gurobi_time_limit))

	print('Done.')

	# The status of the solution is printed to the screen
	print("Status:", LpStatus[prob.status])

	### STEP 6:
	# Retrieve the variable values:

	# Each of the variables is printed with it's resolved optimum value
	for v in prob.variables():
		vval = v.varValue
		if vval > TOLERANCE:
			print(v.name, "=", vval)



	# The optimised objective function value is printed to the screen    

	# Work out the routes:
	all_routes = []
	for v in V:
		print('Nurse: ' + str(v))
		route = [(-1, -1)]
		prev_job = depot
		prev_job_int = -1
		it = 0
		in_route = True
		while in_route and it < nbNodes + 1:
			it += 1
			found_destination = False
			for jint, j in enumerate(C_0):
				for s in S:
					if value(x[prev_job][j][v][s]) > TOLERANCE:
						route.append((jint - 1, s))
						# print('\t', prev_job_int, '\t-\t', jint - 1)
						found_destination = True
						break
				if found_destination:
					prev_job = j
					prev_job_int = jint - 1
					if (prev_job == depot):
						in_route = False
					break
		print('\tRoute: ', route)
		all_routes.append(copy.copy(route))

	print('---------------------')
	print('---------------------')
	
	solMatrix = np.zeros((nbVehi, nbNodes)) - 1
	for n,route in enumerate(all_routes):
		ct = 0
		nstr = 'n_' + str(n)
		print('NURSE ', n)
		for rri in range(len(route) - 1):
			job, skill = route[rri]

			if rri == 0:
				continue

			jstr = 'j_' + str(job + 1)
			sstr = skill

			start_time = value(t[jstr][nstr][sstr])
			tardiness = value(z[jstr][sstr])
			print('\tJOB ', job, ' AT ', start_time, ' with ', tardiness, ' tardiness.')
			solMatrix[n,job] = ct

			ct += 1

	# print('Solmatrix: ')
	# print(str(solMatrix).replace('. ', ', ').replace(']', '],').replace('],],', ']]'))

	obj_value = value(prob.objective)
	print("Obj =\t", obj_value)
	print('D =\t', value(D), '\nD/3 =\t', value(D)/3)
	print('T =\t', value(T), '\nT/3 =\t', value(T)/3)
	print('Tmax =\t', value(Tmax), '\nTmax/3 =\t', value(Tmax)/3)
	return (obj_value, LpStatus[prob.status])

def read_from_matfile(matFile):
    c = scipy.io.loadmat(str(matFile))
    print('Loaded ' + str(matFile))
    nbNodes = c.get('nbNodes')[0][0]
    DS = np.zeros(nbNodes)
    x = np.zeros(nbNodes)
    y = np.zeros(nbNodes)
    xaux = c.get('x')[0]
    DSaux = c.get('DS')[0]
    yaux = c.get('y')[0]
    doDS = True
    for	i in range(nbNodes):
        x[i] = xaux[i][0][0]
        y[i] = yaux[i][0][0]
        if doDS:
            try:
                DS[i] = DSaux[i][0][0]
            except:
                doDS = False
        
    d = c.get('d')
    maxd = c.get('maxd')[0]
    nbVehi = c.get('nbVehi')[0][0]
    r = np.asarray(c.get('r'))
    a = np.asarray(c.get('a'))
    e = np.asarray(c.get('e'))[0]
    l = np.asarray(c.get('l'))[0]
    mind = np.asarray(c.get('mind'))[0]
    nbServi = c.get('nbServi')[0][0]
    pAux = c.get('p')
    p = np.zeros((nbNodes, nbVehi, nbServi))
    ct = 0
    for row,xv in enumerate(pAux):
        for col,v in enumerate(xv):
            ct = ct + 1
            realRow = row % nbVehi
            matx = int(np.floor(float(row) / float(nbVehi)))
            p[matx][realRow][col] = v

    return([nbNodes, nbVehi, nbServi, r, DS, a, x, y, d, p, mind, maxd, e, l])



if __name__ == '__main__':
	print('Running Test for the mankowska_et_al_2014 model...')
	# nbNodes, nbVehi, nbServi, r, DS, a, x, y, d, p, mind, maxd, e, l = MK.read_InstanzCPLEX_HCSRP_10_1()

	directory_mat = r'../data/mankowska_et_al_2014/'
	directory_output = r'../output/'
	# file_list_mat = ['saved_InstanzCPLEX_HCSRP_10_1.mat', 'saved_InstanzCPLEX_HCSRP_10_2.mat', 'saved_InstanzCPLEX_HCSRP_10_3.mat', 'saved_InstanzCPLEX_HCSRP_10_4.mat', 'saved_InstanzCPLEX_HCSRP_10_5.mat', 'saved_InstanzCPLEX_HCSRP_10_6.mat', 'saved_InstanzCPLEX_HCSRP_10_7.mat', 'saved_InstanzCPLEX_HCSRP_10_8.mat', 'saved_InstanzCPLEX_HCSRP_10_9.mat', 'saved_InstanzCPLEX_HCSRP_10_10.mat', 'saved_InstanzCPLEX_HCSRP_25_1.mat', 'saved_InstanzCPLEX_HCSRP_25_2.mat', 'saved_InstanzCPLEX_HCSRP_25_3.mat', 'saved_InstanzCPLEX_HCSRP_25_4.mat', 'saved_InstanzCPLEX_HCSRP_25_5.mat', 'saved_InstanzCPLEX_HCSRP_25_6.mat', 'saved_InstanzCPLEX_HCSRP_25_7.mat', 'saved_InstanzCPLEX_HCSRP_25_8.mat', 'saved_InstanzCPLEX_HCSRP_25_9.mat', 'saved_InstanzCPLEX_HCSRP_25_10.mat']
	file_list_mat = ['saved_InstanzCPLEX_HCSRP_10_1.mat']

	for cfile in file_list_mat:
		matfile = os.path.join(directory_mat, cfile)
		nbNodes, nbVehi, nbServi, r, DS, a, x, y, d, p, mind, maxd, e, l = read_from_matfile(matfile)
		obj_value, status = solve_mk_model(nbNodes, nbVehi, nbServi, r, DS, a, x, y, d, p, mind, maxd, e, l)
		timestamp = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
		f = open(os.path.join(directory_output, 'results_model_mankowska_et_al_2014.txt'), 'a')
		f.write('\n' + str(cfile) + '\t' + str(obj_value) + '\t' + str(status))
		f.close()