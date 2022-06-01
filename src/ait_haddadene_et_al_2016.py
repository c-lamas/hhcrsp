import sys
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import xlrd
import xlsxwriter

from libs.read_helper_functions import read_ait_h_instance
#18 patient set 
filenames=['18-4-s-2a']
# filenames=['18-4-s-2a','18-4-s-2b','18-4-s-2c','18-4-s-2d','18-4-m-2a','18-4-m-2b','18-4-m-2c','18-4-m-2d','18-4-m-2e','18-4-l-2a','18-4-l-2b','18-4-l-2c','18-4-l-2e']

#45 patient set
#filenames=['45-10-s-3a','45-10-s-2a','45-10-s-3b','45-10-s-2b','45-10-s-3c','45-10-m-4','45-10-m-2a','45-10-m-2b','45-10-m-3','45-10-l-2a','45-10-l-2b','45-10-l-3','45-10-l-4']

#75 patient set
#filenames=['73-16-s-2a','73-16-s-3','73-16-s-2b','73-16-m-3a','73-16-m-3b']

workbook = xlsxwriter.Workbook('../output/ait_haddadene_results.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, 'Filename' )
worksheet.write(0, 1, 'Objective value' )
worksheet.write(0, 2, 'Objective Bound' )
worksheet.write(0, 3, 'Runtime' )
worksheet.write(0, 4, 'Preference' )

for fl in range(len(filenames)):
    #Determine the file to read:
    file_to_read = filenames[fl]
    inst = read_ait_h_instance(file_to_read)
    
    # Now you can access this data fields, using a dot:
    # CoutPref - Matrix of the preference scores between patients/carers
    # Demandes - Matrix with skills required by the client
    # DureeDeVisite - Service time per skill -> DureeDeVisite[job][skill]
    # NbClients - Number of patients/clients
    # NbServices - Number of different skills
    # NbVehicles - Number of nurses available
    # Offres - For each nurse, what skills they have. Binary: Offres[nurse][skill]
    # WindowsC - For each job, WindowsC[job][0] is the start of the time window, WindowsC[job][1] the finishing time
    # WindowsD - WindowsD[nurse][0] time the nurse starts working, WindowsD[nurse][1] time the nurse finishes
    # od - od[i,j] as the distance between two patients. row/col 0 represent the nurses location (same for all of them)
    # gap - Gap between two services (for those that require two skills only!) if the gap is 0, it means they are simultaneous
    
    
    class Customer():
        def __init__(self,CID,CPref,SDemand,TWinS,TWinE,SGap):
            self.CID = CID
            self.CPref = CPref
            self.SDemand = SDemand
            self.TWinS = TWinS
            self.TWinE = TWinE
            self.SGap = SGap
        def __str__(self):
            return f"Customer ID: {self.CID}\n Customer Pref: {self.CPref}\n Skill Required: {self.SDemand}\n Time Window Starting: {self.TWinS}\n Time Window Ending: {self.TWinE}\n Gap between two services: {self.SGap}"
    
    
    class Vehicle():
        def __init__(self,VID,STPS,AS,CTWinS,CTWinE):
            self.VID = VID
            self.STPS = STPS
            self.AS = AS
            self.CTWinS=CTWinS
            self.CTWinE=CTWinE
        def __str__(self):
            return f"Vehicle ID: {self.VID}\n Service time per skill: {self.STPS}\n Available Skills: {self.AS}\n Caregiver Availability Starting Time: {self.CTWinS}\n Caregiver Availability Ending Time: {self.CTWinE}"
    
    
    #Read Customer Data
    Customers=[]
    for c in range(inst.NbClients):
        thisCustomer = Customer(c,inst.CoutPref[c],inst.Demandes[c],inst.WindowsC[c][0],inst.WindowsC[c][1],inst.gap[c])
        Customers.append(thisCustomer)
        
    
    #Read Vehicle Data
    Vehicles=[]
    for v in range(inst.NbVehicules):
        thisVehicle = Vehicle(v,inst.DureeDeVisite,inst.Offres,inst.WindowsD[v][0],inst.WindowsD[v][1])
        Vehicles.append(thisVehicle)
        
    
    #Read Location Data
    VLoc = []
    
    for i in range(inst.od.shape[0]):
        thisVLocs = np.hstack((inst.od[i],inst.od[i][0]))
        VLoc.append(thisVLocs) 
    VLoc.append(VLoc[0])
    
    
    def solve_BaseM(Customers,Vehicles):
        V = [i for i in range(inst.NbClients+2)]
        Vnf = [d for d in range(inst.NbClients+1)]
        Vnd = [f for f in range(1,inst.NbClients+2)]
            
        N = [i for i in range(1,inst.NbClients+1)]
        K = [k.VID for k in Vehicles]
        S = [s for s in range(inst.NbServices)]
        
        Ksub = []
        for s in S:
            Ksub.append([])
            for k in K:
                if inst.Offres[k][s] > 0.5:
                    Ksub[s].append(k)
       
        Ssub = []
        for a in range(inst.NbClients):
            Ssub.append([])
            for s in S:
                if inst.Demandes[a][s] > 0.5:
                    Ssub[a].append(s)        
        Ssub.insert(0,[0,1])
        Ssub.append([0,1])
        
        mis = np.array(inst.Demandes)
        
        ##Model
        m = gp.Model("VRP-1")
        
        #Decision Variables
        x = m.addVars(V,V,K,vtype=GRB.BINARY, name="x")
        
        z = m.addVars(N, vtype=GRB.BINARY, name="z")
        
        t = m.addVars(V,K, name="t")
        
        W = m.addVars(V,K, name="W")
        
        #Constraints
        #2
        for k in K:
            m.addConstr(gp.quicksum(x[0,j,k] for j in N) == 1, name="Constraint-2-" + 'k='+ str(k))     
        #3
        for k in K:
            m.addConstr(gp.quicksum(x[i,inst.NbClients+1,k] for i in N) == 1, name="Constraint-3-" + 'k='+ str(k))
        
        #4
        for h in N:
            for k in K:
                m.addConstr((gp.quicksum(x[i,h,k] for i in Vnf) == gp.quicksum(x[h,i,k] for i in Vnd)), name="Constraint-4-"+ 'h=' + str(h) + '_k=' + str(k))
    
        #5:
        for i in N:
            for s in S:
                m.addConstr((gp.quicksum(x[i,j,k] for j in Vnd for k in Ksub[s]) == mis[i-1,s]), name="Constraint-5-"+ 'i=' + str(i) + '_s=' + str(s))
            
        #6    
        for i in V:
            for j in V:
                for s in S:
                    if s in Ssub[i] and s in Ssub[j]: 
                        for k in Ksub[s]:
                            if i==0 or i==len(V)-1:
                                service_time=0
                                end_tw_i=Vehicles[k].CTWinE
                            else:
                                service_time=Vehicles[k].STPS[i-1][s]
                                end_tw_i = Customers[i-1].TWinE
                            m.addConstr(( t[i,k] + (VLoc[i][j] + service_time) * x[(i,j,k)] <= t[j,k] + end_tw_i * (1-x[(i,j,k)])), name="Constraint-6-" + 'i=' + str(i) + '_j=' + str(j) + '_s=' + str(s)+ '_k=' + str(k))    
                   
        #7
        for i in N:
            for s in Ssub[i]:
                for k in Ksub[s]:
                    m.addConstr((Customers[i-1].TWinS * gp.quicksum(x[(i,j,k)] for j in Vnd) <= t[i,k] ), name="Constraint-7L-" + 'i=' + str(i) + '_s=' + str(s)+ '_k=' + str(k))
                    m.addConstr((t[i,k] <= Customers[i-1].TWinE * gp.quicksum(x[(i,j,k)] for j in Vnd) ), name="Constraint-7R-" + 'i=' + str(i) + '_s=' + str(s)+ '_k=' + str(k))   
    
        #8
        for k in K:
            m.addConstr((Vehicles[k].CTWinS <= t[0,k]),name="Constraint-8L-" + 'k='+ str(k))
            m.addConstr((t[0,k] <= Vehicles[k].CTWinE),name="Constraint-8R-" + 'k='+ str(k))
            
        #9
        for k in K:
            m.addConstr( (Vehicles[k].CTWinS <= t[0,k]) ,name="Constraint-9L-" + 'k='+ str(k))
            m.addConstr( (t[0,k] <= Vehicles[k].CTWinE) ,name="Constraint-9R-" + 'k='+ str(k))
            
        GapCons=True
        if GapCons:
            #10
            for i in N:
                for s in Ssub[i]:
                    for r in Ssub[i]:
                        if r<s:
                            m.addConstr((gp.quicksum(t[i,k] for k in Ksub[r])) - (gp.quicksum(t[i,k] for k in Ksub[s])) <= Customers[i-1].SGap ,name="Constraint-10-" + 'i=' + str(i) + '_s=' + str(s)+ '_r=' + str(r))
            #11
            for i in N:
                for s in Ssub[i]:
                    for r in Ssub[i]:
                        if r<s:
                            m.addConstr((gp.quicksum(t[i,k] for k in Ksub[r])) - (gp.quicksum(t[i,k] for k in Ksub[s])) >= -1 * Customers[i-1].SGap ,name="Constraint-11-" + 'i=' + str(i) + '_s=' + str(s)+ '_r=' + str(r))
            #12
            for i in N:
                for s in Ssub[i]:
                    for r in Ssub[i]:
                        if r!=s:
                            m.addConstr((gp.quicksum(t[i,k] for k in Ksub[r])) - (gp.quicksum(t[i,k] for k in Ksub[s])) >= Customers[i-1].SGap - (695*2) * z[i] ,name="Constraint-12-" + 'i=' + str(i) + '_s=' + str(s)+ '_r=' + str(r))            
            #13
            for i in N:
                for s in Ssub[i]:
                    for r in Ssub[i]:
                        if r<s:
                            m.addConstr((gp.quicksum(t[i,k] for k in Ksub[r])) - (gp.quicksum(t[i,k] for k in Ksub[s])) >=  -1 * Customers[i-1].SGap - (695*2) * (1-z[i]) ,name="Constraint-13")                
             
        PrepCons=True
        #14
        if PrepCons:
            for i in N:
                for s in Ssub[i]:
                    if len(Ssub[i])<inst.NbServices and Ssub[i]==Ssub[i+1]:
                        for k in K:
                            if k not in Ksub[s]:
                                m.addConstr(x[i,i+1,k] == 0, name="Constraint-14-" + 'i=' + str(i) + '_s=' + str(s)+ '_k=' + str(k))

        #15    
            for i in N:
                for j in N:
                    for s in Ssub[i]:
                        for k in Ksub[s]:
                            if (Customers[i-1].TWinS + VLoc[i][j] + Vehicles[k].STPS[i-1][s]) >= (Customers[j-1].TWinE + 0.1):
                                m.addConstr(x[i,j,k] == 0, name="Constraint-15-" + 'i=' + str(i) + '_s=' + str(s)+ '_k=' + str(k))
            
        #16
        Con16=True
        if Con16:
            for i in N:
                for s in Ssub[i]:
                    for r in Ssub[i]:
                        if r<s and Customers[i-1].SGap==0:
                            m.addConstr((gp.quicksum(t[i,k] for k in Ksub[r])) == (gp.quicksum(t[i,k] for k in Ksub[s])),name="Constraint-16")         
        #new
        Cons17=False
        if Cons17:
            for i in V:
                for j in V:
                    for s in S:
                        if s in Ssub[i] and s in Ssub[j]: 
                            for k in Ksub[s]:
                                #print(i,j,s,k)
                                if j==0 or j==len(V)-1:
                                    continue
                                else:
                                    service_time=0
                                    end_tw_i=Vehicles[k].CTWinE
                                    end_tw_j=0
                                    if not (i==0 or i==len(V)-1):
                                        service_time=Vehicles[k].STPS[i-1][s]
                                        end_tw_i = Customers[i-1].TWinE
                                        end_tw_j = Customers[j-1].TWinE
                                        m.addConstr(( t[i,k] + (VLoc[i][j] + service_time) * x[(i,j,k)] + W[j,k] <= t[j,k] + end_tw_i * (1-x[(i,j,k)])), name="Constraint-17.1-" + 'i=' + str(i) + '_j=' + str(j) + '_s=' + str(s)+ '_k=' + str(k))    
                                        m.addConstr( W[j,k] >= t[j,k] - (VLoc[i][j] + service_time)* x[(i,j,k)] - (end_tw_j)*(1-x[(i,j,k)]), name="Constraint-17.2-"+ 'i=' + str(i) + '_j=' + str(j) + '_s=' + str(s)+ '_k=' + str(k))
            for i in V:
                for k in K:
                    m.addConstr(W[i,k] >= 0, name="Constraint-18-"+ 'i=' + str(i) + '_k=' + str(k))
    
        
        ##DEBUG_CONSTRAINT WITH OBJ FUNCTION
        #m.addConstr(gp.quicksum(0.3 * VLoc[i][j] * x[i,j,k] for i in Vnf for j in Vnd for k in K) + gp.quicksum(Customers[i-1].CPref[k] * x[i,j,k] for i in N for j in Vnd for k in K)<=220,name="DebugConst")
        
        #Objective Function
        m.setObjective(gp.quicksum(0.3 * VLoc[i][j] * x[i,j,k] for i in Vnf for j in Vnd for k in K) + gp.quicksum(Customers[i-1].CPref[k] * x[i,j,k] for i in N for j in Vnd for k in K), GRB.MINIMIZE)
        m.write("ModellingCode_lp.lp")
        m.Params.timeLimit = 3600.0
        m.optimize()
    
        #m.computeIIS()
        #m.write("ModellingCode_ilp.ilp")
        
        
        route=[]
        Pref=0
        
        for k in K:
            frm=[]
            to=[]
            tym=[]
        
            nurse=[]
            for i in Vnf:
                for j in Vnd:
                    if x[i,j,k].X>0:
                        nurse.append(k)
                        frm.append(i)
                        to.append(j)
                        tym.append(t[j,k].X)
                        if j!=inst.NbClients+1:
                            Pref+=Customers[j-1].CPref[k]
            stck=np.column_stack((nurse,frm,to,tym))
            stck=stck[np.argsort(stck[:,3])]
            route.append(stck)
    
    
        print("****************")
        print("filename = ",file_to_read)
        print("****************")    
        print("Objective Value = ",m.objVal)
        print("****************")
        print("ObjBound=",m.ObjBound)
        print("****************")
        print("Runtime=",m.Runtime)
        print("****************")
        print("Pref=",Pref)
        print("****************")
        
        print(route)
        
        worksheet.write(fl+1, 0, file_to_read )
        worksheet.write(fl+1, 1, m.objVal )
        worksheet.write(fl+1, 2, m.ObjBound )
        worksheet.write(fl+1, 3, m.Runtime )
        worksheet.write(fl+1, 4, Pref )
        
    
    def printScen(scenStr):
        sLen = len(scenStr)
        print("\n" + "*"*sLen + "\n" + scenStr + "\n" + "*"*sLen + "\n")
        
        
    if __name__ == "__main__":
        # Base model
        printScen("Solving base scenario model")
        solve_BaseM(Customers,Vehicles)
workbook.close()     
