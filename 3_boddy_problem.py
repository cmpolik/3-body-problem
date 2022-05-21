# импорт необхадимых библиотек и модулей 
import numpy as np
import scipy as sci
#Import matplotlib для 3D и для анимации
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# гравитационная постоянная
G=6.67408e-11 
# извесные величины
m_nd=1.989e+30 #кг # масса солнца
r_nd=5.326e+12 #м # расстояине м/у свездами системы альфа - центавра
v_nd=30000 #м/с # относительная скорость земли вокруг солнца
t_nd=79.91*365.25*24*3600 #с
# определяемые константы
K1=G*t_nd*m_nd/(r_nd**2*v_nd)
K2=v_nd*t_nd/r_nd

# определяем массы звезд зерез массу солнца
m1=1.1 #Alpha Centauri A
m2=0.907 #Alpha Centauri B
m3=1.425 #Proxima Centauri

# определям начальные положения 
r1=[-0.5,1,0] #m
r2=[0.5,0,0.5] #m
r3=[0.2,1,1.5] #m

# канвертация векторов в списки
r1=np.array(r1)
r2=np.array(r2)
r3=np.array(r3)

# центр масс
r_com=(m1*r1+m2*r2+m3*r3)/(m1+m2+m3)
# начальные скорости
v1=[0.02,0.02,0.02] #m/s
v2=[-0.05,0,-0.1] #m/s
v3=[0,-0.03,0]

# канвертация векторов в списки
v1=np.array(v1)
v2=np.array(v2)
v3=np.array(v3)

# средняя скорость
v_com=(m1*v1+m2*v2+m3*v3)/(m1+m2+m3)

# ф-ия определяющее закон (уравнение) движеняи
def ThreeBodyEquations(w,t,G,m1,m2):
    #расспаковка всех величин из списка "w"
    r1=w[:3]
    r2=w[3:6]
    r3=w[6:9]
    v1=w[9:12]
    v2=w[12:15]
    v3=w[15:18]
    
    # расстояние м/у 3-мя телами
    r12=sci.linalg.norm(r2-r1)
    r13=sci.linalg.norm(r3-r1)
    r23=sci.linalg.norm(r3-r2)
    
    # определяем производные вида dvi/dt 
    dv1bydt=K1*m2*(r2-r1)/r12**3+K1*m3*(r3-r1)/r13**3
    dv2bydt=K1*m1*(r1-r2)/r12**3+K1*m3*(r3-r2)/r23**3
    dv3bydt=K1*m1*(r1-r3)/r13**3+K1*m2*(r2-r3)/r23**3
    
    # dri/dt
    dr1bydt=K2*v1
    dr2bydt=K2*v2
    dr3bydt=K2*v3

    # 6 скалярных дифф. урав. для каждого тела и того всего 18 уравнении
    # загрузим их всех в список размера 18
    r12_derivs=np.concatenate((dr1bydt,dr2bydt))
    r_derivs=np.concatenate((r12_derivs,dr3bydt))
    v12_derivs=np.concatenate((dv1bydt,dv2bydt))
    v_derivs=np.concatenate((v12_derivs,dv3bydt))
    derivs=np.concatenate((r_derivs,v_derivs))
    return derivs


# начальные параметры
init_params=np.array([r1,r2,r3,v1,v2,v3]) # создаем список из начальных параметров 
init_params=init_params.flatten() # сделать из него 1-мерный список
time_span=np.linspace(0,25,500) # !!! временной промежуток составляет 25 орбитальных лет к 500 точам

# решает дифференциальные уравнения
import scipy.integrate
three_body_sol=sci.integrate.odeint(ThreeBodyEquations,init_params,time_span,args=(G,m1,m2))

# распределяем в 3 разных списках
r1_sol=three_body_sol[:,:3]
r2_sol=three_body_sol[:,3:6]
r3_sol=three_body_sol[:,6:9]


# 3D граффик
# создаем точки (тела)
fig=plt.figure(figsize=(15,15))
# 3-мерные оси
ax=fig.add_subplot(111,projection="3d")
# нанесение орбит на график
ax.plot(r1_sol[:,0],r1_sol[:,1],r1_sol[:,2],color="mediumblue")
ax.plot(r2_sol[:,0],r2_sol[:,1],r2_sol[:,2],color="red")
ax.plot(r3_sol[:,0],r3_sol[:,1],r3_sol[:,2],color="gold")
# конечная позиция тел на графике
ax.scatter(r1_sol[-1,0],r1_sol[-1,1],r1_sol[-1,2],color="darkblue",marker="o",s=80,label="Alpha Centauri A")
ax.scatter(r2_sol[-1,0],r2_sol[-1,1],r2_sol[-1,2],color="darkred",marker="o",s=80,label="Alpha Centauri B")
ax.scatter(r3_sol[-1,0],r3_sol[-1,1],r3_sol[-1,2],color="goldenrod",marker="o",s=80,label="Proxima Centauri")
# надписи на осях
ax.set_xlabel("x",fontsize=16)
ax.set_ylabel("y",fontsize=16)
ax.set_zlabel("z",fontsize=16)
ax.set_title("Visualization of orbits of stars in a 3-body system\n",fontsize=14)
ax.legend(loc="upper left",fontsize=14)


# анимация
 
fig=plt.figure(figsize=(15,15))
ax=fig.add_subplot(111,projection="3d")

# создаем новые массивы для анимации
# чтобы уменьшить количество точек в анимации, если она станет медленной
# в настоящее время установлено для выбора каждой 4-й точки
r1_sol_anim=r1_sol[::1,:].copy()
r2_sol_anim=r2_sol[::1,:].copy()
r3_sol_anim=r3_sol[::1,:].copy()

# установ. начальные положения для звезд, то есть синий, красный и зеленый (*) на начальных позициях
head1=[ax.scatter(r1_sol_anim[0,0],r1_sol_anim[0,1],r1_sol_anim[0,2],color="darkblue",marker="*",s=80,label="Alpha Centauri A")]
head2=[ax.scatter(r2_sol_anim[0,0],r2_sol_anim[0,1],r2_sol_anim[0,2],color="darkred",marker="*",s=80,label="Alpha Centauri B")]
head3=[ax.scatter(r3_sol_anim[0,0],r3_sol_anim[0,1],r3_sol_anim[0,2],color="goldenrod",marker="*",s=80,label="Proxima Centauri")]

# ф-ия Animate для каждого i-ого кадра меняет положени точки соотв- решениям другой ф-ий который мы пислаи до 
# т.е. соотв-ии закону движения
def Animate(i,head1,head2,head3):
    #удоление точки для i-1 кадра
    head1[0].remove()
    head2[0].remove()
    head3[0].remove()
    
    # построение орбит (каждую итерацию строится от начального положения до текущего положения)
    trace1=ax.plot(r1_sol_anim[:i,0],r1_sol_anim[:i,1],r1_sol_anim[:i,2],color="mediumblue")
    trace2=ax.plot(r2_sol_anim[:i,0],r2_sol_anim[:i,1],r2_sol_anim[:i,2],color="red")
    trace3=ax.plot(r3_sol_anim[:i,0],r3_sol_anim[:i,1],r3_sol_anim[:i,2],color="gold")
    
    head1[0]=ax.scatter(r1_sol_anim[i-1,0],r1_sol_anim[i-1,1],r1_sol_anim[i-1,2],color="darkblue",marker="*",s=30)
    head2[0]=ax.scatter(r2_sol_anim[i-1,0],r2_sol_anim[i-1,1],r2_sol_anim[i-1,2],color="darkred",marker="*",s=30)
    head3[0]=ax.scatter(r3_sol_anim[i-1,0],r3_sol_anim[i-1,1],r3_sol_anim[i-1,2],color="goldenrod",marker="*",s=30)
    return trace1,trace2,trace3,head1,head2,head3,

# немного приукрашивания
ax.set_xlabel("x-coordinate",fontsize=16)
ax.set_ylabel("y-coordinate",fontsize=16)
ax.set_zlabel("z-coordinate",fontsize=16)
ax.set_title("Visualization of orbits of stars in a 3-body system\n",fontsize=14)
ax.legend(loc="upper left",fontsize=14)


# Использ. модуль FuncAnimation для создания анимации
repeatanim=animation.FuncAnimation(fig,Animate,frames=800,interval=10,repeat=False,blit=False,fargs=(head1,head2,head3))

# ну и запуск всего этого
plt.show()


