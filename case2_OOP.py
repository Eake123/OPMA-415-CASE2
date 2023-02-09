from scipy.optimize import linprog,milp,curve_fit
import pandas as pd
from jsonClass import Json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.animation as animation
LINAC = -426
END_YEAR = 2018
def split_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]
def back_into(x,openb=True):
    s = list(split_list(x,10))
    df = {}
    for c_ind,s1 in enumerate(s):
        c_ind = c_ind + 1
        c = 0
        for count,year in enumerate([2013,2014,2015,2016,2017]):
            if openb:
                open = s1[c]
            else:
                open = s1[c+1]
            c += 2
            v = {c_ind:open}
            if year in df:
                df[year].update(v)
            else:
                df[year] = v
    df = pd.DataFrame(df)
    return df

class ontario:
    def __init__(self,cancer_dict) -> None:
        self.cancer_dict = cancer_dict
        self.forecast_json = Json(r'D:\school\year3-winter\opma415\case2\Forecast.json').readKey()
        # self.forecast = self.create_forecast({x: value for x,value in Json(r'D:\school\year3-winter\opma415\case2\Forecast.json').readKey().items() if x in self.cancer_dict})
        # #print(self.forecast)
        self.forecast = {x: value for x,value in Json(r'D:\school\year3-winter\opma415\case2\Forecast.json').readKey().items() if x in self.cancer_dict}
        self.obj_forecast = self.create_forecast(self.forecast.copy())
        for key,value in self.forecast.items():
            for key2,value2 in self.obj_forecast.items():
                if key == key2:
                    d = {date:v for date,v in value2.items() if int(date) <= END_YEAR}
                value.update(d)
        # pd.DataFrame(self.obj_forecast).plot()
        # plt.show()
        # self.obj_forecast = self.forecast

    def create_forecast(self,forecast):
        new_forecast = ['2018','2019','2020']
        f = {}
        for id,data in forecast.items():

            new_d = {}
            previous_value = None
            for key, value in data.items():
                if previous_value is None:
                    previous_value = value
                else:
                    new_d[key] = value/previous_value
                    previous_value = value
            try:
                params = self.fit_sinusoid(new_d)
                run = True
            except:
                run = False
            if run:
                    keys = list(map(np.float64, new_d.keys()))
                    x = self.sinusoid_values(keys, params)
                    self.plot_sinusoid(new_d,params,id)
                    y = list(new_d.values())

                    regr = DecisionTreeRegressor(random_state=0)
                    regr.fit(x.reshape(-1, 1), y)

                    x_test = self.sinusoid_values(list(new_forecast), params)
                    #print(x_test)
                    y_pred = regr.predict(x_test.reshape(-1, 1))
                    forecast[id] = new_d
                    forecast[id].update(dict(zip(new_forecast, y_pred.tolist())))
            else:
                keys = list(map(np.float64, new_d.keys()))
                x = x = np.array(keys).reshape(-1, 1)
                y = list(new_d.values())

                regr = DecisionTreeRegressor(random_state=0)
                regr.fit(x, y)
                x_test = np.array(list(map(int,list(new_forecast))))
                y_pred = regr.predict(x_test.reshape(-1, 1))
                forecast[id] = new_d
                forecast[id].update(dict(zip(new_forecast, y_pred.tolist())))


            d = {'2013':data['2013']}
            previous_value = None
            for key,value in forecast[id].items():
                if previous_value is None:
                    v = d['2013'] * value
                    previous_value = v
                else:
                    v = previous_value * value
                    previous_value = v
                d[key] = int(v)
            f[id] = d
        return f


    def sinusoid_values(self,keys, params):
        A, omega, phi, b = params
        keys = [np.float64(x) for x in keys]
        return A * np.sin(omega * np.array(keys) + phi) + b
    @staticmethod
    def sinusoid(x, a, b, c, d):
        return a * np.sin(b * x + c) + d
    def fit_sinusoid(self,data):
        x = np.array(list(map(int, data.keys())))
        y = np.array(list(data.values()))
        params, params_covariance = curve_fit(self.sinusoid, x, y)
        return params
    def plot_sinusoid(self,data, params,id):
        x = np.array(list(map(int, data.keys())))
        y = np.array(list(data.values()))
        plt.plot(x, y, 'o', label='data')
        plt.plot(x, self.sinusoid(x, *params), label='fit')
        plt.legend()
        plt.title(f'Sinusoid fit for Cancer Center {id}')
        plt.show()


    def get_objective(self,id):
        obj = self.cancer_dict[id]
        objective = 1 / sum([value for key,value in self.obj_forecast[id].items() if int(key) <= END_YEAR])
        years = [key for key in self.obj_forecast[list(self.obj_forecast)[0]] if int(key) <= END_YEAR]
        l = []
        for id2,data_dict in self.forecast.items():
            #for year in years:
            for year in years:
                if id == id2:
                    empty = objective
                else:
                    empty = 0
                l += [0,empty,0]
        #print(len(l))
        #print(l)
        return l
    
    def get_eq_coefficent(self,id,value):
        l = []
        years = [key for key in self.obj_forecast[list(self.obj_forecast)[0]] if int(key) <= END_YEAR]
        for id1 in self.forecast:
            l2 = []
            for id2,data_dict in self.forecast.items():
                for year in years:
                    if id1 == id2 and id1 == id:
                        v = value
                    else:
                        v = 0
                    l2 += [0,v,0]
            l.append(l2)
    
    def get_id_constraint_coefficent(self):
        l = []
        years = [key for key in self.obj_forecast[list(self.obj_forecast)[0]] if int(key) <= END_YEAR]
        for id1 in self.forecast:
            l2 = []
            for id2,data_dict in self.forecast.items():
                #for year in years:
                for year in years:
                    if id1 == id2:
                        c = self.cancer_dict[id1]
                        empty = 1
                    else:
                        empty = 0
                    l2 += [empty,0,0]
            l.append(l2)
        return l
    
    def get_id_constraint(self):
        l = []
        for id,data_dict in self.forecast.items():
            # for year,value in data_dict.items():
            c = self.cancer_dict[id]
            l.append(c.empty + c.swing)
        return l
    
    def get_year_constraint_coefficent(self):
        l = []
        years = [key for key in self.obj_forecast[list(self.obj_forecast)[0]] if int(key) <= END_YEAR]
        for year in years:
            l2 = []
            for id,data_dict in self.forecast.items():
                for year2 in years:
                    if year == year2:
                        l2 += [1,0,0]
                    else:
                        l2 += [0,0,0]
            l.append(l2)
            

                
        return l
    
    def get_year_constraint(self):
        return [
            4,3,5,2,2,0
        ]

    def get_auo_coefficent(self):
        l = []
        d = {}
        years = [key for key in self.obj_forecast[list(self.obj_forecast)[0]] if int(key) <= END_YEAR]
        for year in years:
            for id1 in self.forecast:
                l2 = []
                for id2,data_dict in self.forecast.items():
                    #for year in years:
                    for year2 in years:
                        if id1 == id2 and year == year2:
                            c = self.cancer_dict[id1]
                            lin = -LINAC
                            v = [lin,1,-1]
                        elif id1 == id2 and int(year) > int(year2):
                            v = [-LINAC,0,0]
                        else:
                            v = [0,0,0]
                        l2 += v
                l.append(l2)
        #print(l)
        return l
    

    
    def get_auo_constraint(self):
        l = []
        years = [key for key in self.obj_forecast[list(self.obj_forecast)[0]] if int(key) <= END_YEAR]
        for year in years:
            for id,data_dict in self.forecast.items():
                for year2,value in data_dict.items():
                    if year == year2:
                        c = self.cancer_dict[id]
                        l.append(value+(c.built * LINAC))
                        # #print(value-c.built)

        #print(l)
        return l
    def get_goal_coefficent(self,id):
        l = []
        d = {}
        years = [key for key in self.obj_forecast[list(self.obj_forecast)[0]] if int(key) <= END_YEAR]
        for id2,data_dict in self.forecast.items():
            #for year in years:
            for year in years:
                if id == id2:
                    empty = 1
                else:
                    empty = 0
                l += [0,empty,0]
        #print(l)
        return [l]
    
                       


                        
                
                


                
            



    


class cancer_centre:
    def __init__(self,id,built=None,e_s=None) -> None:
        self.id = id
        self.built = built if built is not None else Json(r'D:\school\year3-winter\opma415\case2\past.json').readKey()[str(id)]['2013']
        self.e_s = e_s if e_s is not None else Json(r'D:\school\year3-winter\opma415\case2\empty_swing.json').readKey()[str(id)]
        self.swing = self.e_s['Swing Rooms']
        self.empty = self.e_s['Empty Rooms']
    def get_constraint(self,constraint):
        return getattr(self,constraint)

    def __str__(self) -> str:
        return str(self.id)
    
    def get(self):
        return [self.empty,self.swing]




class optimize:
    def __init__(self,cancer_dict) -> None:
        self.cancer_dict = cancer_dict
        self.o = ontario(cancer_list)
    
    def get_A(self):
        A = self.o.get_year_constraint_coefficent() + self.o.get_id_constraint_coefficent() #+ self.o.get_auo_coefficent()        #print(self.o.get_auo_coefficent())
        # #print()
        # #print(len(self.o.get_year_constraint_coefficent()))
        #print(len(self.o.get_auo_coefficent()))
        # #print(len(self.o.get_id_constraint_coefficent()))
        # #print(self.o.get_id_constraint_coefficent())
        return A

    def get_A_eq(self):
        return self.o.get_auo_coefficent()
    def get_b(self):
        b = self.o.get_year_constraint()  + self.o.get_id_constraint() #+ self.o.get_auo_constraint()
        #print(len(self.o.get_year_constraint()))
        #print(len(self.o.get_auo_constraint()))
        #print(len(self.o.get_id_constraint()))
        return b
    
    def get_b_eq(self):
        return self.o.get_auo_constraint()
    def get_c(self,id):
        c = self.o.get_objective(id)
        return c

    def linprog(self,id,A=None,b=None,c=None,A_eq=None,b_eq=None):
        A = A if A is not None else self.get_A()
        A_eq = A_eq if A_eq is not None else self.get_A_eq()
        b_eq = b_eq if b_eq is not None else self.get_b_eq()
        b = b if b is not None else self.get_b()
        c = c if c is not None else self.get_c(id)
        # #print(len(A),len(b))
        # #print(self.back_into(A_eq[27]))
        # #print(A_eq[12],b_eq[12])
        res = linprog(c,A_ub=A,b_ub=b,A_eq=A_eq,b_eq=b_eq,integrality=[1 for x in range(len(c))])
        return res
    def callback(xk, **kwargs):
        print("Current solution:", xk)
    def get_all_constraints(self):
        s = self.get_sorted()
        res = self.linprog('1')
        # print(len(self.o.get_year_constraint_coefficent()[0]))
        l = {}
        dfs = []
        for count,id in enumerate(s):
            id = str(id)
            if count == 0:
                res = self.linprog(id)
                print(id)
                new_constraint = self.make_new_constraint(res.x,int(id))
                goal_co = self.o.get_goal_coefficent(id)
                goal_b = [new_constraint]
                l[id] = res.fun
            else:
                A_eq = self.get_A_eq() + goal_co
                b_eq = self.get_b_eq() + goal_b
                res = self.linprog(id,A_eq=A_eq,b_eq=b_eq)
                l[id] = res.fun
                # res = self.linprog(id)
                new_constraint = self.make_new_constraint(res.x,int(id))
                goal_co += self.o.get_goal_coefficent(id)
                    # print(self.o.get_goal_coefficent(id))
                goal_b += [new_constraint]
                    # print(id)
            dfs.append(self.create_dfs(res.x,s))
        # self.animate_it(dfs)
        self.animate_built(res.x,s)
    
    def animate_built(self,x,s):
        total_built = {}
        j = Json(r'D:\school\year3-winter\opma415\case2\past.json').readKey()
        years = [2005+x for x in range(2021-2005)]
        df = {}
        tw = True
        th = True
        for id in s:
            d = j[id]
            for year in years:
                year = str(year)
                s = 0
                if year in d:
                    s += d[year]
                    if year == '2012' and tw:
                        s += 1
                        tw = False
                    if year == '2013' and th:
                        s += 1
                        th = False
                    already_built = s
                else:
                    s = already_built
                new = self.get_built_after(x,int(id),year)  
                    
                already_built += new    
                          
                if year in df:
                    df[year]['Already Built'] += s
                    df[year]['Newly Built'] += new
                else:
                    df[year] = {'Already Built' : s}
                    df[year].update({'Newly Built' : new})
                    
        df = pd.DataFrame(df).transpose()
        print(df)
            # data = df.iloc[i:i+1, :]
        def update(num):
            ax.clear()
            df_year = df[df.index <= str(years[num])]
            df_year.plot(kind='bar', stacked=True, ax=ax)
            ax.set_ylim(60, ax.get_ylim()[1])
            plt.title('Total new Cancer Centers Built per Year')
            plt.xlabel('Year')
            plt.ylabel('Built')
        fig, ax = plt.subplots()
        # df.plot(kind='bar',stacked=True)
        ani = animation.FuncAnimation(fig, update, frames=len(years), repeat=False)
        plt.show()
        ani.save('animated_stacked_bar.gif', writer='imagemagick')
                                
    
    def get_built_after(self,x,id_num,year):
        year = int(year)
        df = self.back_into(x,id_num)
        if year in list(df.columns):
            return df[year]['rooms']
        return 0

    def animate_it(self,df_list):
        def animate(i):
            data = df_list[i]
            plt.clf()
            # data.plot()
            plt.title('Cancer Center % of Patients Without Linac')
            plt.xlabel('Year')
            plt.ylabel('% of Patients Without Linac')
            plt.ylim(0, 0.7)
            for column in data.columns:
                plt.plot(data[column],label=column)

            plt.legend(loc='upper left')
            if i == len(df_list) - 1:
                ani.event_source.stop()


        fig = plt.figure()
        ani = animation.FuncAnimation(fig, animate, frames=len(df_list), interval=200)
        ani.save("patients_no_linac.gif", writer="imagemagick")
        plt.show()
            
        
    def create_dfs(self,x,s):
        old_df = {}
        total = 'total'
        for id in s:
            old_df[id] = self.make_chart_part(x,int(id))
        df = {}
        for id,data_dict in old_df.items():
            for year,value in data_dict.items():
                if total in df:
                    if year in df[total]:
                        df[total][year] += value
                    else:
                        df[total][year] = value
                else:
                    df[total]= {year:value}
        old_df.update(df)
        df = old_df

        total_df = df[total].copy()
        df[total] = {date:value / len(s) for date,value in total_df.items()} 
        df = pd.DataFrame(df)
        return df
        
    
    def make_chart_part(self,x,id_num):
        df = self.back_into(x,id_num)
        l = {}
        
        for col in df.columns:
            v = df[col]['under'] / self.o.obj_forecast[str(id_num)][str(col)]
            l[col] = v

        return l




    def make_new_constraint(self,x,id_num):
        df = self.back_into(x,id_num)
        s = 0
        for col in df.columns:
            s += df[col]['under']
        return s




    def get_sorted(self):
        df = {}
        for id,data in self.o.forecast.items():
            df[id] = data['2017']
        df = [x[0] for x in sorted(df.items(), key=lambda x: x[1], reverse=True)]
        return df
    def animate(self):
        def update(num):
            # Add new data to the animation
            x.append(num)
            y.append(abs(res.fun))

            # Clear the previous plot
            ax.clear()

            # Plot the new data
            ax.plot(x, y)

            # Set the labels and title for the plot
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Objective function value')
            ax.set_title('Linprog optimization iterations')
        fig, ax = plt.subplots()

        # Initialize the data for the animation
        x = [0]
        y = [0]
        res = self.linprog()
        ani = FuncAnimation(fig, update, frames=range(1,res.nit+1), repeat=False)
        plt.show()

    def change_constraint(self,id,l):
        # years = ['2013','2014','2015','2016','2017']

        l[int(id)-1] += 1
    
    def compare(self):
        df = {str(x):[] for x in range(1,17)}
        for id in self.cancer_dict:
            first = self.o.get_year_constraint()
            second = self.o.get_id_constraint('empty')
            third = self.o.get_id_constraint('swing')
            for i in range(20):
                self.change_constraint(id,second)
                res = self.linprog(b=first + second + third)
                df[id].append(res.fun)
        df = pd.DataFrame(df)
        df.plot()
        plt.show()
    
        
        
    def plot_built(self):
        built = pd.DataFrame(Json(r'D:\school\year3-winter\opma415\case2\past.json').readKey())
        x = self.linprog().x
        # open_room = {col:row[col] for col,row in zip(self.back_into(x).columns,self.back_into(x).index)}
        open_room = self.sum_year(x)
        swing_room = self.sum_year(x,False)

        built = {date:x for x,date in zip(built['Total'],built.index)}
        years = set([int(x) for x in list(built.keys()) + list(open_room.keys())])
        years = sorted(years)
        df = {'Already Built':[],'Open Room':[],'Swing Room':[]}
        open_total = 0
        swing_total = 0
        for year in years:
            year = str(year)
            if year in built:
                df['Already Built'].append(built[year])
                last_year = built[year]
            else:
                df['Already Built'].append(last_year)
            if year in open_room:
                open_total += open_room[year]
                df['Open Room'].append(open_total)
            else:
                df['Open Room'].append(0)
            if year in swing_room:
                swing_total += swing_room[year]
                df['Swing Room'].append(swing_total)
            else:
                df['Swing Room'].append(0)
        def update(num):
            ax.clear()
            df_year = df[df.index <= years[num]]
            df_year.plot(kind='bar', stacked=True, ax=ax)
            ax.set_ylim(60, ax.get_ylim()[1])
        df = pd.DataFrame(df,index=years)
        fig, ax = plt.subplots()
        # df.plot(kind='bar',stacked=True)
        ani = animation.FuncAnimation(fig, update, frames=len(years), repeat=False)
        ani.save('animated_stacked_bar.gif', writer='imagemagick')
        # plt.ylim(60,plt.ylim()[1])
        plt.show()
    


    def sum_year(self,x,openb=True):
        df = {}
        df_first = self.back_into(x,openb)
        for col in df_first.columns:
            df[str(col)] = sum(df_first[col])   
        return df

    @staticmethod
    def split_list(input_list, chunk_size):
        return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]
    def back_into(self,x,id_num):
        #print(len(x))
        s = list(self.split_list(x,18))
        df = {}
        id = 1
        for c_ind,s1 in enumerate(s):
            c_ind = c_ind + 1
            c = 0
            for count,year in enumerate([2013,2014,2015,2016,2017,2018]):
                # room_key = f'rooms {id} {year}'
                # under_key = f'under {id} {year}'
                # over_key = f'over {id} {year}'
                if id == id_num:
                    v = {'rooms':s1[c],'under':s1[c+1],'over':s1[c+2]}
                    # if openb:
                    #     open = s1[c]
                    # else:
                    #     open = s1[c+1]
                    c += 3
                    # v = {c_ind:open}
                    if year in df:
                        df[year].update(v)
                    else:
                        df[year] = v
            id += 1
        df = pd.DataFrame(df)
        return df
    def get_eq_constraint(self,x):
        l = {}
        id = 1
        x = self.split_list(x,6)
        for i in x:
            for count,year in enumerate([2013,2014,2015,2016,2017,2018]):
                v = {year:i[count]}
                if id in l:
                    l[id].update(v)
                else:
                    l[id] = v
            id += 1
        return pd.DataFrame(l).transpose()





cancer_list = {str(x):cancer_centre(x) for x in range(1,17)}
o = optimize(cancer_list)
# o.plot_built()
# o = ontario(cancer_list)
# #print(o.get_objective('13'))
# #print(o.get_objective('13'))
# #print(o.linprog('13').x)
# #print(o.back_into(o.linprog('13').x))
print(o.get_all_constraints())
# #print(o.get_c())
# #print(o.back_into(o.linprog().x,True))
# #print(o.get_c())

# o = ontario(cancer_list)


# #print(cancer_list)
# o = ontario(cancer_list)




# #print(back_into(c))
# #print(back_into(res.x))
# #print(back_into(res.x,False))
# #print(res.fun)
# my_list = [i for i in range(160)]









# #print([x * y for x,y in zip(res.x,o.get_objective())])



# d = o.get_year_constraint_coefficent()

# #print(d[0] == d[10])
# # o = ontario([])
# # #print(o.get_bounds())



# Initialize the data for the animation

