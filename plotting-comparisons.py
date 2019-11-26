import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# original training loss (ADAM)
training_losses_original_lr001 = [0.5561568729934239, 0.12334978802218324, 0.0767864462963882, 0.05527657711701024, 0.046438158378892, 0.03525828916047301, 0.028554040705785155, 0.02165116462856531, 0.02014924113505653, 0.020345278350370272]
training_losses_original_lr01 = [0.7379901132413319, 0.09990987487669502, 0.07092657020049435, 0.0543764941837816, 0.043846919356534876, 0.03201405758348604, 0.03148124776115375, 0.038830860140955166, 0.02909517024887637, 0.03309062977011005]
# baseline training loss (ADAM)
training_losses_baseline_lr01 = [0.36530595490088064, 0.07839477270664204, 0.04658891956898428, 0.03962309691788895, 0.03387203430091696, 0.024615694261488637, 0.025101072218690422, 0.02628867021606614, 0.02404643671997335, 0.019864573972188822]
training_losses_baseline_lr001 = [0.682210682935658, 0.1502913668574322, 0.09790222643918935, 0.07842442042948235, 0.06082370380560557, 0.053672560158052614, 0.04431722257729797, 0.038883017730854806, 0.03552161140500435, 0.029969510988199284]
training_losses_baseline_lr05 = [2.7570954646383012, 2.302611932868049, 2.3022023921921138, 2.3024537648473467, 2.3027218466713313, 2.302331016177223, 2.302284586997259, 2.302561109974271, 2.3025860587755838, 2.3029464937391735]
training_losses_baseline_lr005 = [0.4022251667366141, 0.07916118515034516, 0.04634342568793467, 0.037745371516350476, 0.027899868292955772, 0.023403509209553402, 0.019840647078429658, 0.01444478287538957, 0.009924928014654489, 0.015856885492602096]
#training loss (SGD)
training_losses_SGD_lr001 = [2.3021186874026345, 2.301759787968227, 2.301397136279515, 2.301030022757394, 2.300657405739739, 2.3002775254703702, 2.299889635472071, 2.2994938691457114, 2.299088472411746, 2.298673362958999]
training_losses_SGD_lr01 = [2.300480794338953, 2.2962407157534646, 2.290558922858465, 2.281510208334242, 2.263983209927877, 2.216254257020496, 1.9699505936531794, 0.9970644144784837, 0.5915212666704541, 0.4507146431576638]
training_losses_SGD_lr005 = [2.301408554826464, 2.2995033065478006, 2.297387134461176, 2.2950117815108526, 2.2922285482996987, 2.288757074446905, 2.284313834848858, 2.278459784530458, 2.2701247249330794, 2.257212323801858]
training_losses_SGD_lr05 = [2.287263206073216, 1.4794464327749752, 0.4087891497072719, 0.2601029856928757, 0.19985602281632878, 0.16889286271872975, 0.13511509687772819, 0.11547599874791645, 0.10428018184999625, 0.08881355964002155]
training_losses_SGD_lrPoint1 = [2.09556905854316, 0.5070708335510322, 0.2094486134038085, 0.1436196949687742, 0.1257860455662012, 0.11101728597921985, 0.08279241688017334, 0.06703201805551846, 0.058566803201323465, 0.05069680514168881]
# validation acc vs. time (ADAM)
time_adam_lr01 = [0.1670382022857666, 0.33307576179504395, 0.496112585067749, 0.6641504764556885, 0.8231866359710693, 0.9727277755737305, 1.1297638416290283, 1.28779935836792, 1.4418342113494873, 1.598869800567627, 1.7599060535430908, 1.9199423789978027, 2.080981731414795, 2.245161533355713, 2.407198429107666, 2.5762362480163574, 2.741274356842041, 2.9103121757507324, 3.0973544120788574, 3.262392044067383, 3.423428535461426, 3.589466094970703, 3.744478940963745, 3.907747745513916, 4.071805000305176, 4.235841751098633, 4.3928186893463135, 4.556856155395508, 4.709894418716431, 4.8699305057525635, 5.0329749584198, 5.185009241104126, 5.347046136856079, 5.5130839347839355, 5.681060314178467, 5.840111255645752]
time_adam_lr001 = [0.11402606964111328, 0.21704936027526855, 0.3190724849700928, 0.4300975799560547, 0.5371220111846924, 0.6451461315155029, 0.7491698265075684, 0.8561937808990479, 0.9592177867889404, 1.0632424354553223, 1.167264461517334, 1.2762889862060547, 1.3943159580230713, 1.5103421211242676, 1.615365982055664, 1.7233905792236328, 1.8334157466888428, 1.9434418678283691, 2.070469379425049, 2.1804940700531006, 2.2885186672210693, 2.3995437622070312, 2.5115694999694824, 2.6125922203063965, 2.717615842819214, 2.8256404399871826, 2.929664134979248, 3.0416901111602783, 3.1587159633636475, 3.2677500247955322, 3.3766908645629883, 3.4933931827545166, 3.6004176139831543, 3.711442708969116, 3.8254685401916504, 3.9294919967651367]
time_adam_lr05 = [0.11402654647827148, 0.22205018997192383, 0.33707642555236816, 0.448101282119751, 0.5641279220581055, 0.6751527786254883, 0.7881786823272705, 0.9022040367126465,
1.0162303447723389, 1.1342570781707764, 1.25028395652771, 1.354306697845459, 1.4683327674865723, 1.5783581733703613, 1.6903831958770752, 1.811410903930664, 1.9174349308013916, 2.0304601192474365, 2.1544880867004395, 2.260512351989746, 2.369537115097046, 2.4885647296905518, 2.5965890884399414, 2.7096142768859863, 2.8126373291015625, 2.9256627559661865, 3.0356881618499756, 3.151715040206909, 3.2672438621520996, 3.3802480697631836, 3.4892725944519043, 3.6042990684509277, 3.719324827194214, 3.8333518505096436, 3.946376323699951, 4.060402870178223]
time_adam_lr005 = [0.1225285530090332, 0.2483971118927002, 0.37032628059387207, 0.49041152000427246, 0.6120791435241699, 0.7344374656677246, 0.855381965637207, 0.97967529296875, 1.1050844192504883, 1.2270009517669678, 1.3522164821624756, 1.478930950164795, 1.6036956310272217, 1.726701259613037, 1.8504881858825684, 1.9745380878448486, 2.098475933074951, 2.2187509536743164, 2.3397908210754395, 2.4619150161743164, 2.581451177597046, 2.700944185256958, 2.8214964866638184, 2.942044734954834, 3.0649189949035645, 3.185633659362793, 3.307199716567993, 3.4365804195404053, 3.581368923187256, 3.729163408279419, 3.8757400512695312, 4.011024475097656, 4.132397413253784, 4.268056154251099, 4.391629457473755, 4.512211799621582]

validation_acc_adam_lr01 = [98.8, 98.5, 98.6, 98.55, 98.6, 98.5, 98.48571428571428, 98.575, 98.57777777777778, 98.58, 98.49090909090908, 98.5, 98.47692307692307, 98.54285714285714, 98.56, 98.575, 98.54117647058824, 98.58888888888889, 98.61052631578947, 98.57, 98.56190476190476, 98.58181818181818, 98.59130434782608, 98.58333333333333, 98.584, 98.56153846153846, 98.56296296296296, 98.57142857142857, 98.55862068965517, 98.55333333333333, 98.52903225806452, 98.54375, 98.50909090909092, 98.51764705882353, 98.50285714285714, 98.52222222222223]
validation_acc_adam_lr001 = [98.2, 98.3, 98.26666666666667, 98.4, 98.4, 98.3, 98.28571428571429, 98.375, 98.4, 98.38, 98.38181818181818, 98.46666666666667, 98.46153846153847, 98.52857142857142, 98.56, 98.6, 98.55294117647058, 98.57777777777778, 98.57894736842105, 98.57, 98.54285714285714, 98.54545454545455, 98.52173913043478, 98.525, 98.528, 98.50769230769231, 98.52592592592593, 98.55, 98.54482758620689, 98.54, 98.54838709677419, 98.5375, 98.52121212121212, 98.52352941176471, 98.53142857142858, 98.54444444444445]
validation_acc_adam_lr05 = [12.4, 11.0, 10.733333333333333, 10.45, 10.44, 10.533333333333333, 10.714285714285714, 11.0, 10.8, 11.04, 10.89090909090909, 10.85, 10.892307692307693, 10.928571428571429, 11.013333333333334, 10.95, 10.929411764705883, 11.033333333333333, 11.115789473684211, 11.12, 11.17142857142857, 11.272727272727273, 11.321739130434782,
11.416666666666666, 11.376, 11.3, 11.296296296296296, 11.292857142857143, 11.427586206896551, 11.373333333333333, 11.393548387096773, 11.35625, 11.284848484848485, 11.25294117647059, 11.222857142857142, 11.25]
validation_acc_adam_lr005 = [98.2, 98.1, 98.46666666666667, 98.65, 98.52, 98.5, 98.54285714285714, 98.625, 98.57777777777778, 98.52, 98.54545454545455, 98.61666666666666, 98.56923076923077, 98.62857142857143, 98.64, 98.675, 98.67058823529412, 98.68888888888888, 98.67368421052632, 98.67, 98.66666666666667, 98.68181818181819, 98.69565217391305, 98.7, 98.696, 98.7, 98.71111111111111, 98.72857142857143, 98.73103448275862, 98.73333333333333, 98.73548387096774, 98.7375, 98.73333333333333, 98.73529411764706, 98.74857142857142, 98.76666666666667]
# validation acc vs. time (SGD)
time_SGD_lr01 = [0.12883639335632324, 0.25879430770874023, 0.3859226703643799, 0.5301258563995361, 0.6663999557495117, 0.7921712398529053, 0.9180326461791992, 1.044370412826538, 1.1709225177764893, 1.296950340270996, 1.4232661724090576, 1.5513134002685547, 1.6822025775909424, 1.808361291885376, 1.9344677925109863, 2.0606229305267334, 2.1866135597229004, 2.3122715950012207, 2.44313383102417, 2.587697982788086, 2.714205265045166, 2.861288547515869, 3.0055298805236816, 3.1491103172302246, 3.292196750640869, 3.4181103706359863, 3.5439248085021973, 3.672724962234497, 3.799098253250122, 3.925422191619873, 4.051208734512329, 4.178037166595459, 4.325149059295654, 4.4524171352386475, 4.580040693283081, 4.713048696517944]
time_SGD_lr001 = [0.11702775955200195, 0.23105144500732422, 0.3490784168243408, 0.4671049118041992, 0.5901334285736084, 0.7191627025604248, 0.8391897678375244, 0.9572169780731201, 1.0752432346343994, 1.1932697296142578, 1.3122975826263428, 1.4323241710662842, 1.5593533515930176, 1.6833817958831787, 1.841416835784912, 1.9694459438323975, 2.0974748134613037, 2.210500955581665, 2.3405301570892334, 2.4535558223724365, 2.570582151412964, 2.6766059398651123, 2.7786290645599365, 2.881653070449829, 2.9916775226593018, 3.105703115463257, 3.2137277126312256, 3.327754020690918, 3.4387786388397217, 3.5488040447235107, 3.663830041885376, 3.7708547115325928, 3.875877857208252,
3.9829020500183105, 4.093927383422852, 4.206953048706055]
time_SGD_lr005 = [0.14590954780578613, 0.27615904808044434, 0.4029831886291504, 0.5308136940002441, 0.6579174995422363, 0.8027598857879639, 0.9367885589599609, 1.080385684967041, 1.2085154056549072, 1.3346004486083984, 1.460904598236084, 1.5877244472503662, 1.71583890914917, 1.84163498878479, 1.9692821502685547, 2.1140248775482178, 2.2576701641082764, 2.401198148727417, 2.545814037322998, 2.6905956268310547, 2.8315463066101074, 2.9582924842834473, 3.085604429244995, 3.212932825088501, 3.339036703109741, 3.465425729751587, 3.5916483402252197, 3.7192397117614746, 3.8567843437194824, 4.001314878463745, 4.146107196807861, 4.289509534835815, 4.433613061904907, 4.578318119049072, 4.722495079040527, 4.856376886367798]
time_SGD_lr05 = [0.13086915016174316, 0.263486385345459, 0.39154553413391113, 0.5160748958587646, 0.6381099224090576, 0.760617733001709, 0.8832828998565674, 1.006622076034546, 1.1283581256866455, 1.2498135566711426, 1.3726255893707275, 1.4953055381774902, 1.6170997619628906, 1.7390954494476318, 1.8622210025787354, 1.9856293201446533, 2.107590436935425, 2.229872703552246, 2.3521006107330322, 2.4799132347106934, 2.603739023208618, 2.7261219024658203, 2.850524425506592, 2.9746084213256836, 3.097365140914917, 3.2199270725250244, 3.3422937393188477, 3.4651167392730713, 3.588987112045288, 3.7140190601348877, 3.84503173828125, 3.981127977371216, 4.1035637855529785, 4.22614860534668, 4.348467111587524, 4.471530914306641]
time_SGD_lrPoint1 = [0.11002516746520996, 0.21504902839660645, 0.330075740814209, 0.4471018314361572, 0.5511250495910645, 0.669152021408081, 0.7861783504486084, 0.8952028751373291, 1.004227876663208, 1.1182537078857422, 1.229278802871704,
1.3333024978637695, 1.4373259544372559, 1.5423498153686523, 1.6553752422332764, 1.7744028568267822, 1.8894286155700684, 1.9984533786773682, 2.115914821624756, 2.219938278198242, 2.3249623775482178, 2.429989814758301, 2.5429203510284424, 2.6599466800689697, 2.7679712772369385, 2.8799967765808105, 2.9900217056274414, 3.106048345565796, 3.2139220237731934, 3.3229222297668457, 3.4249460697174072, 3.536104440689087, 3.6493542194366455, 3.7633800506591797, 3.869403839111328, 3.9844303131103516]

validation_acc_SGD_lr01 = [86.8, 85.7, 86.33333333333333, 86.25, 85.88, 86.0, 85.71428571428571, 85.75, 85.5111111111111, 85.24, 85.05454545454545, 85.0, 85.03076923076924, 85.15714285714286, 85.13333333333334, 85.075, 85.21176470588236, 85.23333333333333, 85.2, 85.19, 85.23809523809524, 85.37272727272727, 85.31304347826087, 85.43333333333334, 85.472, 85.45384615384616, 85.32592592592593, 85.37857142857143, 85.35862068965517, 85.3, 85.29032258064517, 85.21875, 85.24242424242425, 85.25294117647059, 85.25714285714285, 85.28888888888889]
validation_acc_SGD_lr001 = [9.6, 10.2, 9.466666666666667, 10.25, 10.32, 10.433333333333334, 10.371428571428572, 10.05, 10.133333333333333, 9.96, 10.0, 9.966666666666667, 9.923076923076923, 9.957142857142857, 9.96, 9.9125, 9.870588235294118, 9.844444444444445, 9.726315789473684, 9.69, 9.666666666666666, 9.736363636363636, 9.704347826086957, 9.625, 9.576, 9.569230769230769, 9.548148148148147, 9.542857142857143, 9.558620689655172, 9.593333333333334, 9.63225806451613, 9.6375, 9.684848484848485, 9.676470588235293, 9.674285714285714, 9.65]
validation_acc_SGD_lr005 = [53.2, 53.0, 51.86666666666667, 51.9, 52.08, 52.6, 51.94285714285714, 51.85, 51.55555555555556, 51.32, 51.36363636363637, 51.333333333333336, 51.56923076923077, 51.52857142857143, 51.54666666666667, 51.675, 51.76470588235294, 51.8, 51.705263157894734, 51.48, 51.63809523809524, 51.96363636363636, 52.0, 52.19166666666667, 51.976, 52.05384615384615, 51.925925925925924, 51.92857142857143, 51.91724137931035, 51.92, 51.86451612903226, 51.8625, 51.82424242424243, 51.794117647058826, 51.794285714285714, 51.75]
validation_acc_SGD_lr05 = [96.8, 96.6, 96.73333333333333, 96.8, 96.64, 96.66666666666667, 96.74285714285715, 96.925, 96.82222222222222, 96.66, 96.6, 96.71666666666667, 96.73846153846154, 96.85714285714286, 96.82666666666667, 96.85, 96.84705882352941, 96.87777777777778, 96.88421052631578, 96.89, 96.81904761904762, 96.82727272727273, 96.80869565217391, 96.775, 96.784, 96.76923076923077, 96.71851851851852, 96.74285714285715, 96.70344827586207, 96.66, 96.69677419354839, 96.6875, 96.67878787878787, 96.68235294117648, 96.66285714285715, 96.67777777777778]
validation_acc_SGD_lrPoint1 = [97.2, 97.3, 97.46666666666667, 97.4, 97.44, 97.43333333333334, 97.45714285714286, 97.575, 97.46666666666667, 97.4, 97.36363636363636, 97.45, 97.46153846153847, 97.6, 97.56, 97.6125, 97.6, 97.61111111111111, 97.63157894736842, 97.64, 97.61904761904762, 97.62727272727273, 97.6086956521739, 97.60833333333333, 97.632, 97.63846153846154, 97.60740740740741, 97.63571428571429, 97.57241379310345, 97.52666666666667, 97.54838709677419, 97.55, 97.53939393939395, 97.55294117647058, 97.52, 97.55555555555556]

# Plotting training loss at epoch num
def plot_training_loss(epochs, training_losses_original, training_losses_baseline, lr):
    plt.plot(epochs, training_losses_original, marker='o', linestyle='dashed', label='Original')
    plt.plot(epochs, training_losses_baseline, marker="o", linestyle="dashed", label='Baseline')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs. Epoch (lr=(%.4f))' % lr)
    plt.legend()
    plt.show()

def plot_validation_accuracy(time_array, validation_acc_original, validation_acc_baseline):
    time = []
    validation_accuracy_original_lr01 = []
    # Plots validation accuracy over time
    plt.plot(time, validation_acc_original)
    plt.xlabel('Time (s)')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs. Time')
    plt.show()

plot_training_loss(epochs, training_losses_original_lr01, training_losses_baseline_lr01, 0.01)
plot_training_loss(epochs, training_losses_original_lr001, training_losses_baseline_lr001, 0.001)

# Plotting training loss vs. epoch ( ADAM LR comparison)
plt.plot(epochs, training_losses_baseline_lr001, marker='o', linestyle='dashed', label='0.001')
plt.plot(epochs, training_losses_baseline_lr005, marker="o", linestyle="dashed", label='0.005')
plt.plot(epochs, training_losses_baseline_lr01, marker="o", linestyle="dashed", label='0.01')

plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. Epoch (ADAM LR comparisons)')
plt.legend()
plt.show()

# Plotting valid accuracy vs. time ( ADAM LR comparisons)
plt.plot(time_adam_lr001, validation_acc_adam_lr001, label='0.001')
plt.plot(time_adam_lr005, validation_acc_adam_lr005, label='0.005')
plt.plot(time_adam_lr01, validation_acc_adam_lr01, label='0.01')

plt.xlabel('Time')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy vs. Time (ADAM LR comparisons)')
plt.legend()
plt.show()

# Plotting training loss vs. epoch ( SGD LR comparison)
plt.plot(epochs, training_losses_SGD_lr001, marker='o', linestyle='dashed', label='0.001')
plt.plot(epochs, training_losses_SGD_lr005, marker="o", linestyle="dashed", label='0.005')
plt.plot(epochs, training_losses_SGD_lr01, marker="o", linestyle="dashed", label='0.01')
plt.plot(epochs, training_losses_SGD_lr05, marker="o", linestyle="dashed", label='0.05')
plt.plot(epochs, training_losses_SGD_lrPoint1, marker="o", linestyle="dashed", label='0.1')

plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. Epoch (SGD LR comparisons)')
plt.legend()
plt.show()

# Plotting valid accuracy vs. time ( SGD LR comparisons)
# plt.plot(time_SGD_lr001, validation_acc_SGD_lr001, label='0.001')
# plt.plot(time_SGD_lr005, validation_acc_SGD_lr005, label='0.005')
# plt.plot(time_SGD_lr01, validation_acc_SGD_lr01, label='0.01')
plt.plot(time_SGD_lr05, validation_acc_SGD_lr05, label='0.05')
plt.plot(time_SGD_lrPoint1, validation_acc_SGD_lrPoint1, label='0.1')

plt.xlabel('Time')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy vs. Time (SGD LR comparisons)')
plt.legend()
plt.show()

# PROBLEM 2b (adam vs sgd) -----------------
# training loss vs epoch
plt.plot(epochs, training_losses_baseline_lr005, marker='o', linestyle="dashed", label='ADAM: LR=0.005')
plt.plot(epochs, training_losses_SGD_lrPoint1, marker='o', linestyle="dashed", label='SGD: LR=0.1')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. Epochs (Adam vs. SGD)')
plt.legend()
plt.show()

# validation acc vs. time
plt.plot(time_adam_lr005, validation_acc_adam_lr005, label='SGD: LR=0.1')
plt.plot(time_SGD_lrPoint1, validation_acc_SGD_lrPoint1, label='ADAM: LR=0.005')
plt.xlabel('Time (s)')
plt.ylabel('Validation Accuracy')
plt.title('Validation Acc vs. Time (Adam vs. SGD)')
plt.legend()
plt.show()

# activation functions (part 2.2)
time_adam_sigmoid_lr005 = [0.1180267333984375, 0.23505353927612305, 0.3430781364440918, 0.4631052017211914, 0.5781311988830566, 0.6971578598022461, 0.8191854953765869, 0.9328222274780273, 1.04780912399292, 1.1598188877105713, 1.2798209190368652, 1.3898155689239502, 1.496811866760254, 1.6088156700134277, 1.7178151607513428, 1.8258123397827148, 1.9358241558074951, 2.0508241653442383, 2.1818149089813232, 2.2907416820526123, 2.410816192626953, 2.529817581176758,
2.637812614440918, 2.7508246898651123, 2.8708083629608154, 2.9828052520751953, 3.0968217849731445, 3.2018179893493652, 3.311814546585083, 3.431812047958374, 3.5498170852661133, 3.654808282852173, 3.761779308319092, 3.8799140453338623, 3.9858222007751465, 4.092822074890137]
validation_acc_adam_sigmoid_lr005 = [95.8, 95.6, 96.06666666666666, 95.95, 95.88, 95.93333333333334, 95.88571428571429, 96.05, 95.95555555555555, 95.82, 95.83636363636364, 95.86666666666666, 95.87692307692308, 95.9, 95.94666666666667, 95.9375, 95.94117647058823, 95.95555555555555, 96.0, 95.97, 95.94285714285714, 95.95454545454545, 95.88695652173914, 95.96666666666667, 95.968, 95.92307692307692, 95.86666666666666, 95.89285714285714, 95.87586206896552, 95.83333333333333, 95.83870967741936, 95.8125, 95.7939393939394, 95.81176470588235, 95.78285714285714, 95.82222222222222]
plt.plot(time_adam_sigmoid_lr005, validation_acc_adam_sigmoid_lr005, label="Sigmoid.Adam.lr=0.005")
plt.plot(time_adam_lr005, validation_acc_adam_lr005, label="ReLU.Adam.lr=0.005")
plt.xlabel('Time (s)')
plt.ylabel('Validation Accuracy')
plt.title('Validation Acc vs. Time (Sigmoid vs. ReLU)')
plt.legend()
plt.show()