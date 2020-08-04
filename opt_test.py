import os
import shutil
import platform
import numpy as np
import pyemu


bin_path = os.path.join("test_bin")
if "linux" in platform.platform().lower():
    bin_path = os.path.join(bin_path,"linux")
elif "darwin" in platform.platform().lower():
    bin_path = os.path.join(bin_path,"mac")
else:
    bin_path = os.path.join(bin_path,"win")

bin_path = os.path.abspath("test_bin")
os.environ["PATH"] += os.pathsep + bin_path


# case of either appveyor, travis or local
if os.path.exists(os.path.join("pestpp","bin")):
    bin_path = os.path.join("..","..","pestpp","bin")
else:
    bin_path = os.path.join("..","..","..","..","pestpp","bin")
    
if "windows" in platform.platform().lower():
    exe_path = os.path.join(bin_path, "win", "pestpp-opt.exe")
elif "darwin" in platform.platform().lower():
    exe_path = os.path.join(bin_path,  "mac", "pestpp-opt")
else:
    exe_path = os.path.join(bin_path, "linux", "pestpp-opt")


def std_weights_test():
    d = os.path.join("opt_dewater_chance", "test_std_weights")
    if os.path.exists(d):
        shutil.rmtree(d)
    shutil.copytree(os.path.join("opt_dewater_chance", "template"), d)
    pst_file = os.path.join(d, "dewater_pest.base.pst")
    jco_file = os.path.join(d, "dewater_pest.full.jcb")
    pst = pyemu.Pst(pst_file)
    par = pst.parameter_data
    par.loc[par.partrans == "fixed", "partrans"] = "log"
    jco = pyemu.Jco.from_binary(jco_file)
    par.loc[par.pargp == "q", "partrans"] = "fixed"
    obs = pst.observation_data.loc[pst.nnz_obs_names, :]

    forecast_names = list(obs.loc[obs.obgnme.apply(lambda x: x.startswith("l_") or \
                                                             x.startswith("less_")), "obsnme"])
    # print(forecast_names)
    pst.observation_data.loc[:, "weight"] = 0.0
    sc = pyemu.Schur(jco=jco, pst=pst, forecasts=forecast_names)
    # print(sc.get_forecast_summary())

    fstd = sc.get_forecast_summary().loc[:, "post_var"].apply(np.sqrt)
    pr_unc_py = fstd.to_dict()
    pst.observation_data.loc[fstd.index, "weight"] = fstd.values

    pst.pestpp_options["opt_risk"] = 0.1
    pst.pestpp_options["opt_std_weights"] = True
    pst.pestpp_options["base_jacobian"] = os.path.split(jco_file)[-1]
    par.loc[par.pargp == "q", "partrans"] = "none"

    new_pst_file = os.path.join(d, "test.pst")

    pst.write(new_pst_file)
    pyemu.os_utils.run("{0} {1}".format(exe_path, os.path.split(new_pst_file)[-1]), cwd=d)
    pr_unc1 = scrap_rec(new_pst_file.replace(".pst", ".rec"))

    print(pr_unc1)
    for fore in forecast_names:
        dif = np.abs(pr_unc_py[fore] - pr_unc1[fore])
        print(fore, pr_unc_py[fore], pr_unc1[fore], dif)
        assert dif < 1.0e-4

    pst.pestpp_options["opt_std_weights"] = False
    pst.write(new_pst_file)
  
    pyemu.os_utils.run("{0} {1}".format(exe_path, os.path.split(new_pst_file)[-1]), cwd=d)
    pr_unc2 = scrap_rec(new_pst_file.replace(".pst", ".rec"))
    print(pr_unc2)
    for fore in forecast_names:
        dif = np.abs(pr_unc_py[fore] - pr_unc2[fore])
        print(fore, pr_unc_py[fore], pr_unc2[fore], dif)
        assert dif < 1.0e-4,dif


def scrap_rec(rec_file):
    unc = {}
    tag = "FOSM-based chance constraint information at start of iteration 1"
    # this alt tag is to support mou dev 
    tag_alt = "FOSM-based chance constraint/objective information at start of iteration 1"
    with open(rec_file, 'r') as f:
        while True:
            line = f.readline()
            if line == "":
                break

            if tag in line or tag_alt in line:
                f.readline()
                while True:
                    line = f.readline()
                    if line == "":
                        break
                    raw = line.strip().split()
                    # print(raw)
                    try:
                        name = raw[0].lower()
                        val = float(raw[4])
                        unc[name] = val
                    except:
                        break
                break
    return unc


def run_dewater_test():
    worker_d = os.path.join("opt_dewater_chance")
    pyemu.os_utils.start_workers(os.path.join(worker_d, "template"), exe_path, "dewater_pest.base.pst",
                                master_dir=os.path.join(worker_d, "master"), worker_root=worker_d, num_workers=10,
                                verbose=True,port=4200)

    opt = None
    with open(os.path.join(worker_d, "master", "dewater_pest.base.rec"), 'r') as f:
        for line in f:
            if "iteration 1 objective function value:" in line:
                opt = float(line.strip().split()[-2])
    assert opt is not None

    pst = pyemu.Pst(os.path.join(worker_d,"template","dewater_pest.base.pst"))
    pst.control_data.noptmax = 3
    pst.write(os.path.join(worker_d,"template","test.pst"))
    pyemu.os_utils.start_workers(os.path.join(worker_d, "template"), exe_path, "test.pst",
                                master_dir=os.path.join(worker_d, "master"), worker_root=worker_d, num_workers=10,
                                verbose=True,port=4200)
    
    with open(os.path.join(worker_d,"master","test.rec")) as f:
        for line in f:
            if "iteration       obj func" in line:
                f.readline() # skip the initial obj func
                lines = []
                for _ in range(pst.control_data.noptmax):
                    lines.append(f.readline())
    obj_funcs = np.array([float(line.strip().split()[-1]) for line in lines])
    print(obj_funcs)
    assert np.abs(obj_funcs.max() - obj_funcs.min()) < 0.1

    pst.pestpp_options["opt_risk"] = 0.95
    pst.pestpp_options.pop("base_jacobian",None)
    par = pst.parameter_data
    adj_pars = par.loc[par.pargp=="h","parnme"]
    pst.parameter_data.loc[adj_pars[0],"partrans"] = "log"
    pst.parameter_data.loc[adj_pars[1:],"partrans"] = "tied"
    pst.parameter_data.loc[adj_pars[1:],"partied"] = adj_pars[0]
    pst.parameter_data.loc[["up_grad","dn_grad"],"partrans"] = "log"
    pst.control_data.noptmax = 1
    pst.write(os.path.join(worker_d,"template","test.pst"))
    pyemu.os_utils.start_workers(os.path.join(worker_d, "template"), exe_path, "test.pst",
                                master_dir=os.path.join(worker_d, "master"), worker_root=worker_d, num_workers=10,
                                verbose=True,port=4200)
    with open(os.path.join(worker_d,"master","test.rec")) as f:
        for line in f:
            if "iteration       obj func" in line:
                f.readline() # skip the initial obj func
                lines = []
                for _ in range(pst.control_data.noptmax):
                    lines.append(f.readline())
    averse_obj_funcs = np.array([float(line.strip().split()[-1]) for line in lines])
    print(averse_obj_funcs) 
    



def run_supply2_test():
    worker_d = os.path.join("opt_supply2_chance")
    pyemu.os_utils.start_workers(os.path.join(worker_d, "template"), exe_path, "supply2_pest.base.pst",
                                master_dir=os.path.join(worker_d, "master"), worker_root=worker_d, num_workers=10,
                                verbose=True,port=4200)

    opt = None
    with open(os.path.join(worker_d, "master", "supply2_pest.base.rec"), 'r') as f:
        for line in f:
            if "iteration 1 objective function value:" in line:
                opt = float(line.strip().split()[-2])
    assert opt is not None

    pst = pyemu.Pst(os.path.join(worker_d,"template","supply2_pest.base.pst"))
    pst.control_data.noptmax = 10
    pst.pestpp_options["opt_iter_tol"] = 1.0e-10
    pst.write(os.path.join(worker_d,"template","test.pst"))
    pyemu.os_utils.start_workers(os.path.join(worker_d, "template"), exe_path, "test.pst",
                                master_dir=os.path.join(worker_d, "master"), worker_root=worker_d, num_workers=10,
                                verbose=True,port=4200)
    
    with open(os.path.join(worker_d,"master","test.rec")) as f:
        for line in f:
            if "iteration       obj func" in line:
                f.readline() # skip the initial obj func
                lines = []
                for _ in range(pst.control_data.noptmax):
                    lines.append(f.readline())
    obj_funcs = np.array([float(line.strip().split()[-1]) for line in lines])
    print(obj_funcs)
    #assert np.abs(obj_funcs.max() - obj_funcs.min()) < 0.1




def est_res_test():
    worker_d = os.path.join("opt_supply2_chance")
    t_d = os.path.join(worker_d,"template")
    m_d = os.path.join(worker_d,"master")
    if os.path.exists(m_d):
        shutil.rmtree(m_d)

    pst = pyemu.Pst(os.path.join(t_d,"supply2_pest.base.pst"))
    obs = pst.observation_data
    obs.loc[:,"weight"] = 1.0
    pst.pestpp_options["opt_risk"] = 0.05
    pst.pestpp_options["opt_std_weights"] = True
    pst.write(os.path.join(t_d,"supply2_pest.base.pst"))
    pyemu.os_utils.start_workers(os.path.join(worker_d, "template"), exe_path, "supply2_pest.base.pst",
                                master_dir=os.path.join(worker_d, "master"), worker_root=worker_d, num_workers=10,
                                verbose=True,port=4200)

    opt = None
    with open(os.path.join(worker_d, "master", "supply2_pest.base.rec"), 'r') as f:
        for line in f:
            if "iteration 1 objective function value:" in line:
                opt = float(line.strip().split()[-2])
    assert opt is not None
    res_est1 = pyemu.pst_utils.read_resfile(os.path.join(m_d,"supply2_pest.base.1.est+chance.rei"))

    for f in ["supply2_pest.base.1.jcb","supply2_pest.base.1.jcb.rei"]:
        shutil.copy2(os.path.join(m_d,f),os.path.join(t_d,f))
    pst = pyemu.Pst(os.path.join(t_d,"supply2_pest.base.pst"))
    pst.pestpp_options["opt_skip_final"] = True
    pst.pestpp_options["base_jacobian"] = "supply2_pest.base.1.jcb"
    pst.pestpp_options["hotstart_resfile"] = "supply2_pest.base.1.jcb.rei"
    pst.write(os.path.join(t_d,"pest_est_res.pst"))
    m_d = os.path.join(worker_d,"master_est_res")
    if os.path.exists(m_d):
        shutil.rmtree(m_d)
    shutil.copytree(t_d,m_d)
    pyemu.os_utils.run("{0} pest_est_res.pst".format(exe_path),cwd=m_d)
    res_est2 = pyemu.pst_utils.read_resfile(os.path.join(m_d,"pest_est_res.1.est+chance.rei"))
    diff = (res_est1.modelled - res_est2.modelled).apply(np.abs)
    print(diff.sum())
    assert diff.sum() < 1.0e-10

def stack_test():
    d = os.path.join("opt_dewater_chance", "stack_test")
    if os.path.exists(d):
        shutil.rmtree(d)
    shutil.copytree(os.path.join("opt_dewater_chance", "template"), d)
    pst_file = os.path.join(d, "dewater_pest.base.pst")
    pst = pyemu.Pst(pst_file)
    par = pst.parameter_data
    par.loc[par.partrans=="fixed","partrans"] = "log"
    pst.pestpp_options["opt_risk"] = 0.1
    pst.pestpp_options["opt_stack_size"] = 10
    pst.control_data.noptmax = 1
    new_pst_file = os.path.join(d, "test.pst")

    pst.write(new_pst_file)
    pyemu.os_utils.run("{0} {1}".format(exe_path, os.path.split(new_pst_file)[-1]), cwd=d)
    rec1 = os.path.join(d,"test.rec")
    assert os.path.exists(rec1)
    par_stack = "test.1.par_stack.csv"
    assert os.path.exists(os.path.join(d,par_stack))
    shutil.copy2(os.path.join(d,par_stack),os.path.join("opt_dewater_chance", "template","par_stack.csv"))
    pst.pestpp_options["opt_par_stack"] = "par_stack.csv"
    
    d = d + "_par_stack"
    if os.path.exists(d):
        shutil.rmtree(d)
    shutil.copytree(os.path.join("opt_dewater_chance", "template"), d)
    pst.write(os.path.join(d,"test.pst"))
    pyemu.os_utils.run("{0} {1}".format(exe_path, "test.pst"), cwd=d)
    rec2 = os.path.join(d,"test.rec")
    assert os.path.exists(rec2)
    par_stack = "test.1.par_stack.csv"
    assert os.path.exists(os.path.join(d,par_stack))
    obs_stack = "test.1.obs_stack.csv"
    assert os.path.exists(os.path.join(d,obs_stack))
    shutil.copy2(os.path.join(d,obs_stack),os.path.join("opt_dewater_chance", "template","obs_stack.csv"))
    shutil.copy2(os.path.join(d,par_stack),os.path.join("opt_dewater_chance", "template","par_stack.csv"))
    pst.pestpp_options["opt_obs_stack"] = "obs_stack.csv"
    pst.pestpp_options["opt_par_stack"] = "par_stack.csv"
    
    d = d + "_obs_stack"
    if os.path.exists(d):
        shutil.rmtree(d)
    shutil.copytree(os.path.join("opt_dewater_chance", "template"), d)
    pst.write(os.path.join(d,"test.pst"))
    pyemu.os_utils.run("{0} {1}".format(exe_path, "test.pst"), cwd=d)   
    rec3 = os.path.join(d,"test.rec")
    assert os.path.exists(rec3)

    def get_obj(rec_file):
        tag = "---  objective function sequence  ---"
        with open(rec_file,'r') as f:
            for line in f:
                if tag in line:
                    for _ in range(3):
                        line = f.readline()
                    #print(line)
                    obj = float(line.strip().split()[-1])
                    print(obj)
                    return obj

    obj1 = get_obj(rec1)
    obj2 = get_obj(rec2)
    obj3 = get_obj(rec3)
    assert np.abs(obj1 - obj2) < 1.0e-1
    assert np.abs(obj2 - obj3) < 1.0e-1

    d = os.path.join("opt_dewater_chance","stack_iter_test")
    if os.path.exists(d):
        shutil.rmtree(d)
    pst.control_data.noptmax = 5
    pst.pestpp_options["opt_recalc_chance_every"] = 1
    pst.pestpp_options["opt_stack_size"] = 30
    pst.pestpp_options.pop("opt_par_stack")
    pst.pestpp_options.pop("opt_obs_stack")
    
    pst.write(os.path.join("opt_dewater_chance","template","test.pst"))
    pyemu.os_utils.start_workers(os.path.join("opt_dewater_chance", "template"), exe_path, "test.pst",
                                master_dir=d, worker_root="opt_dewater_chance", num_workers=10,
                                verbose=True,port=4200)


def dewater_restart_test():
    worker_d = os.path.join("opt_dewater_chance")
    pst = pyemu.Pst(os.path.join(worker_d,"template","dewater_pest.base.pst"))
    par = pst.parameter_data
    par.loc[par.partrans=="fixed","partrans"] = "log"
    pst.pestpp_options.pop("base_jacobian",None)
    pst.control_data.noptmax = 1
    pst.pestpp_options.pop("opt_risk",None)
    pst.write(os.path.join(worker_d,"template","base.pst"))
    pyemu.os_utils.start_workers(os.path.join(worker_d, "template"), exe_path, "base.pst",
                                master_dir=os.path.join(worker_d, "master_base1"), worker_root=worker_d, num_workers=10,
                                verbose=True,port=4200)

    
    pst.control_data.noptmax = 1
    shutil.copy2(os.path.join(worker_d,"master_base1","base.1.jcb"),os.path.join(worker_d,"template","restart.jcb"))
    pst.pestpp_options["base_jacobian"] = "restart.jcb"
    pst.write(os.path.join(worker_d,"template","restart.pst"))
    pyemu.os_utils.run("{0} restart.pst".format(exe_path),cwd=os.path.join(worker_d,"template"))
    
    with open(os.path.join(worker_d,"master","test.rec")) as f:
        for line in f:
            if "iteration       obj func" in line:
                f.readline() # skip the initial obj func
                lines = []
                for _ in range(pst.control_data.noptmax):
                    lines.append(f.readline())
    obj_funcs = np.array([float(line.strip().split()[-1]) for line in lines])
    print(obj_funcs)
    assert np.abs(obj_funcs.max() - obj_funcs.min()) < 0.1

    

if __name__ == "__main__":
    std_weights_test()
    #run_dewater_test()
    #run_supply2_test()
    # est_res_test()
    #stack_test()
    #dewater_restart_test()
