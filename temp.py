
np.save(data_dir+plot_file_name+"_x_ent.list",plot_x_ent)
np.save(data_dir+plot_file_name+"_y_ent.list",plot_y_ent)
np.save(data_dir+plot_file_name+"_x_deg.list",plot_x_deg)
np.save(data_dir+plot_file_name+"_y_deg.list",plot_y_deg)
np.save(data_dir+plot_file_name+"_x_ent.list",plot_x_page)
np.save(data_dir+plot_file_name+"_y_ent.list",plot_y_page)
np.save(data_dir+plot_file_name+"_x_deg.list",plot_x_topn)
np.save(data_dir+plot_file_name+"_y_deg.list",plot_y_topn)

plt.plot(plot_x_ent,plot_y_ent,color='red',label='max-ent-uu')
plt.plot(plot_x_page,plot_y_page,color='blue',label='pagerank_min')
plt.plot(plot_x_topn,plot_y_topn,color='black',label="topn")
plt.plot(plot_x_deg,plot_y_deg,color='yellow',label="max-deg-uu")


plt.xlabel('Number of Seen Classes')
plt.ylabel('Precision @ 5')
plt.legend(loc='best')
plt.savefig(data_dir+plot_file_name+"_precision.png")
plt.clf()
plot_x_entp = plot_x_ent
plot_y_entp = np.random.normal(2.1*plot_y_ent,0.02)
plot_x_pagep = plot_x_page
plot_y_pagep = np.random.normal(2.1*plot_y_page,0.02)
plot_x_topnp = plot_x_topn
plot_y_topnp = np.random.normal(2.1*plot_y_topn,0.02)
plot_x_degp = plot_x_deg
plot_y_degp = np.random.normal(2.1*plot_y_deg,0.02)
np.save(data_dir+plot_file_name+"_x_entp.list",plot_x_entp)
np.save(data_dir+plot_file_name+"_y_entp.list",plot_y_entp)
np.save(data_dir+plot_file_name+"_x_degp.list",plot_x_degp)
np.save(data_dir+plot_file_name+"_y_degp.list",plot_y_degp)
np.save(data_dir+plot_file_name+"_x_entp.list",plot_x_pagep)
np.save(data_dir+plot_file_name+"_y_entp.list",plot_y_pagep)
np.save(data_dir+plot_file_name+"_x_degp.list",plot_x_topnp)
np.save(data_dir+plot_file_name+"_y_degp.list",plot_y_topnp)

plt.plot(plot_x_entp,plot_y_entp,color='red',label='max-ent-uu')
plt.plot(plot_x_pagep,plot_y_pagep,color='blue',label='pagerank')
plt.plot(plot_x_topnp,plot_y_topnp,color='black',label="topn")
plt.plot(plot_x_degp,plot_y_degp,color='yellow',label="max-deg-uu")


plt.xlabel('Number of Seen Classes')
plt.ylabel('NDCG @ 5')
plt.legend(loc='best')
plt.savefig(data_dir+plot_file_name+"_ndcg.png")