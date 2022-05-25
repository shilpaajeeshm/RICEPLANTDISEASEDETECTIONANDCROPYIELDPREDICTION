import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from flask import Flask, render_template, request, session
from DBConnection import Db
app = Flask(__name__)
app.secret_key="hii"

@app.route('/home')
def home():
    return render_template('admin/admin_index.html')





@app.route('/log')
def login():
    return render_template('login_index.html')



@app.route('/login_post',methods=["post"])
def login_post():
    uname = request.form["textfield"]
    password = request.form["textfield3"]
    db3 = Db()
    ser ="SELECT * FROM login WHERE Email='"+uname+"' AND PASSWORD='"+password+"' "
    rd=db3.selectOne(ser)
    if rd!='' and rd is not None:
        session['lid']=rd['L_id']
        type=rd['TYPE']
        if type=='admin':
            return '''<script>alert('success');window.location='/home'</script>'''
        elif type=='user':
            db5 = Db()
            sdf ="SELECT * FROM user WHERE lid='"+str(rd["L_id"])+"'"
            rd1=db5.selectOne(sdf)
            if rd1 is not None:
                session["uimg"]=rd1["Photo"]
                return '''<script>alert('success');window.location='/profile'</script>'''
            else:
                return '''<script>alert('Invalid');window.location='/'</script>'''

        else:
            return '''<script>alert('Invalid');window.location='/'</script>'''
    else:
        return '''<script>alert('Invalid');window.location='/'</script>'''


@app.route('/change_password')
def change_admin_password():
    return render_template('admin/change_admin_password.html')

@app.route('/change_admin_password_post',methods=["post"])
def change_admin_password_post():
    current_password = request.form["textfield"]
    new_password = request.form["textfield2"]
    confirm_password = request.form["textfield30"]
    db=Db()
    qry="SELECT * FROM login WHERE password='"+current_password+"' AND L_id='"+str(session['lid'])+"'"
    res=db.selectOne(qry)
    if (res!=None):
        if (new_password==confirm_password):
            qry1="update login set password='"+new_password+"' where  L_id='"+str(session['lid'])+"'"
            res1=db.update(qry1)
            return '''<script>alert('password changed');window.location='/'</script>'''
        else :
            return '''<script>alert('password not changed');window.location='/change_password'</script>'''
    else :
        return '''<script>alert('current password not matching');window.location='/change_password'</script>'''



@app.route('/user_change_password')
def user_change_password():
    return render_template('user/change_password.html')

@app.route('/user_change_password_post',methods=["post"])
def user_change_password_post():
    current_password = request.form["textfield"]
    new_password = request.form["textfield2"]
    confirm_password = request.form["textfield3"]
    db=Db()
    qry="SELECT * FROM login WHERE password='"+current_password+"'"
    res=db.selectOne(qry)
    if (new_password==confirm_password):
        qry1="update login set password='"+new_password+"' where  L_id='"+str(session['lid'])+"'"
        res1=db.update(qry1)
        return '''<script>alert('password changed');window.location='/log'</script>'''
    else :
        return '''<script>alert('password not changed');window.location='/user_change_password'</script>'''


@app.route('/registration')
def register():
    return render_template('registration_index.html')
@app.route('/register_post',methods=["post"])
def register_post():
    uname=request.form["textfield4"]
    phone_no=request.form["textfield5"]
    email = request.form["textfield6"]
    password=request.form["textfield7"]
    confirm_password=request.form["textfield8"]
    place = request.form["place"]
    image = request.files["file"]
    image.save("C:\\Users\\user\\PycharmProjects\\riceplantdisase\\static\\userimg\\"+image.filename)
    url='/static/userimg/'+image.filename
    d=Db()

    q="INSERT INTO login(Email,PASSWORD,TYPE)VALUES('"+email+"','"+confirm_password+"','user');"
    lid=d.insert(q)


    qry="INSERT INTO USER(phone_no,NAME,lid,Photo,Email,place)VALUES('"+phone_no+"','"+uname+"','"+str(lid)+"','"+url+"','"+email+"','"+place+"')"
    d.insert(qry)
    return '''<script>alert('success');window.location='/'</script>'''



@app.route('/forget_password')
def forget():
    return render_template('forgettemplate.html')

@app.route('/forget_post',methods=["post"])
def forget_post():
    email=request.form["textfield10"]
    db6 = Db()
    fgh = "SELECT * FROM login WHERE Email='"+email+"'"
    res = db6.selectOne(fgh)
    x=res["PASSWORD"]
    # import random
    # x=random.randint(0000,9999)
    if res is not None:
        s = smtplib.SMTP(host='smtp.gmail.com', port=587)
        s.starttls()
        s.login("riceplantleafdiseasedetection@gmail.com", "rice@123")
        msg = MIMEMultipart()  # create a message.........."
        message = "Messege from rice hub"
        msg['From'] = "riceplantleafdiseasedetection@gmail.com"
        msg['To'] = email
        msg['Subject'] = "Your Password for rice hub "
        body = "Your Account has been verified by our team. You Can login using your password - " + str(x)
        msg.attach(MIMEText(body, 'plain'))
        s.send_message(msg)
        return '''<script>alert("send");window.location='/log'</script>'''
    else:
        return '''<script>alert("Invalid");window.location='/forget_password'</script>'''

@app.route('/crop_yield')
def crop_yield():
    return render_template('user/crop_yield.html')
@app.route('/crop_yield',methods=["post"])
def crop_yield_post():
    Rain_fall=request.form["textfield12"]
    Humidity = request.form["textfield14"]
    Soil_pH=request.form["textfield15"]
    Temperature = request.form["textfield13"]
    import pandas as pd
    # from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    tp = float(Temperature)
    hd = float(Humidity)
    ph = float(Soil_pH)
    rf = float(Rain_fall)
    a = [[tp, hd, ph, rf]]
    data = pd.read_csv(
        "C:\\Users\\user\\PycharmProjects\\riceplantdisase\\static\\predictiondata.csv")
    atribute = data.iloc[1:, 0:4].values
    label = data.iloc[1:, 4].values
    # from sklearn.tree import DecisionTreeClassifier
    # c=DecisionTreeClassifier()
    # c.fit(atribute,label)
    # p=c.predict(a)
    # neigh = KNeighborsClassifier(n_neighbors=4)
    # neigh.fit(atribute, label)
    # # KNeighborsClassifier(...)
    # a = neigh.predict(a)
    # print("pppppp---" + a[0])
    # resulit = str(a[0])


    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100)

    rf.fit(atribute, label)

    predictedresult = rf.predict(a)

    return render_template('user/crop_yield.html',res=predictedresult[0])
    # return a[0]




@app.route('/disease_detection')
def disease_detection():
    return render_template('user/disease_detection.html')
@app.route('/show_post',methods=["post"])
def disease_detection_post():
    image = request.files["file"]

    image.save("C:\\Users\\user\\PycharmProjects\\riceplantdisase\\static\\check_img\\"+image.filename)
    url = '/static/userimg/'+image.filename
    qxy= "INSERT INTO DISEASE_DETECTION(Photo,user_L_id,date)VALUES('"+url+"','"+str(session['lid'])+"',curdate())"
    dt = Db()
    dt.insert(qxy)

    import numpy as np
    from skimage import io, color, img_as_ubyte
    import pandas as pd

    from skimage.feature import greycomatrix, greycoprops
    from sklearn.metrics.cluster import entropy
    path = 'C:\\Users\\user\\PycharmProjects\\riceplantdisase\\static\\check_img\\'+image.filename

    rgbImg = io.imread(path)
    grayImg = img_as_ubyte(color.rgb2gray(rgbImg))

    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    properties = ['energy', 'homogeneity','dissimilarity', 'correlation', 'contrast']

    glcm = greycomatrix(grayImg,
                        distances=distances,
                        angles=angles,
                        symmetric=True,
                        normed=True)

    feats = np.hstack([greycoprops(glcm, 'homogeneity').ravel()
                       for prop in properties])
    feats1 = np.hstack([greycoprops(glcm, 'energy').ravel()
                        for prop in properties])
    feats2 = np.hstack(
        [greycoprops(glcm, 'dissimilarity').ravel() for prop in properties])
    feats3 = np.hstack(
        [greycoprops(glcm, 'correlation').ravel() for prop in properties])
    feats4 = np.hstack([greycoprops(glcm, 'contrast').ravel()
                        for prop in properties])

    k = np.mean(feats)
    l = np.mean(feats1)
    m = np.mean(feats2)
    n = np.mean(feats3)
    o = np.mean(feats4)

    aa = [k, l, m, n, o]

    df = pd.read_csv('C:\\Users\\user\\PycharmProjects\\riceplantdisase\\static\\image_detection1.csv')
    attributes = df.values[:, 1:6]
    # print(len(attributes))
    label = df.values[:, 6]
    str(df)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        attributes, label, test_size=0.1, random_state=42)

    from sklearn.ensemble import RandomForestClassifier
    a = RandomForestClassifier(n_estimators=100)

    a.fit(X_train, y_train)

    predictedresult = a.predict([aa])
    resulit= str(predictedresult[0])

    d = Db()
    res="SELECT * FROM adminadd WHERE D_name like '%"+resulit+"%' "

    da=d.selectOne(res)
    # print(da)
    # print(res)
    if da is not None:
        return render_template('user/disease_detection.html',res=resulit,symptoms=da["Symptom"],cause=da["Cause"],remedy=da["Remedy"])
    else:
        return render_template('user/disease_detection.html',res=resulit)


@app.route('/view_profile')
def view_profile():
    db4 = Db()
    q = "SELECT * FROM user WHERE lid='"+str(session['lid'])+"'"
    res = db4.selectOne(q)

    return render_template('user/view_profile.html',data=res)


@app.route('/edit_user_pofile/<id>')
def edit_user_pofile(id):
    d = Db()
    qry = "SELECT * FROM `user` WHERE lid='"+str(id)+"'"
    res = d.selectOne(qry)
    return render_template("user/edit_user_profile.html", data=res)

@app.route('/edit_user_pofile_post', methods=['post'])
def edit_user_pofile_post():
    uname = request.form['textfield54']
    phone_no = request.form['textfield55']
    email = request.form['textfield58']
    place = request.form['textfield56']

    if 'file' in request.files:
        image = request.files['file']
        # print("---------------------")
        if image.filename!='':
            # print("---------------------****")
            image.save("C:\\Users\\user\\PycharmProjects\\riceplantdisase\\static\\userimg\\" + image.filename)
            url = '/static/userimg/' + image.filename
            db=Db()
            qry = "UPDATE user SET `phone_no`='"+phone_no+"',`NAME`='"+uname+"',`Photo`='"+url+"',`Email`='"+email+"',`place`='"+place+"' where lid='"+str(session["lid"])+"'"
            db.update(qry)
            return '''<script>alert('success');window.location='/view_profile'</script>'''
        else:
            # print("---------------------1")
            db = Db()
            qry = "UPDATE user SET `phone_no`='"+phone_no+"',`NAME`='"+uname+"',`Email`='"+email+"',`place`='"+place+"' where lid='"+str(session["lid"])+"'"
            db.update(qry)
            return '''<script>alert('success');window.location='/view_profile'</script>'''
    else:
        # print("---------------------2")
        db = Db()
        qry = "UPDATE user SET `phone_no`='"+phone_no+"',`NAME`='"+uname+"',`Email`='"+email+"',`place`='"+place+"' where lid='"+str(session["lid"])+"'"
        db.update(qry)
        return '''<script>alert('success');window.location='/view_profile'</script>'''

    return '''<script>alert('success');window.location='/view_profile'</script>'''


@app.route('/profile')
def profile():

    return render_template('user/user_index.html')



@app.route('/rating')
def rating():
    return render_template('user/rating.html')

@app.route('/submit_post',methods=["post"])
def submit_post():
    Rating = request.form["select"]
    db7 = Db()
    qry = "INSERT INTO `rating`(`L_id`,`Rating`,`Date`) VALUES('"+str(session['lid'])+"','"+Rating+"',curdate())"
    db7.insert(qry)

    return rating()




@app.route('/admin_add_disease')
def admin_add():
    return render_template('admin/admin_add_disease.html')
@app.route('/admin_add_post',methods=["post"])
def admin_add_post():

    Name = request.form["textfield17"]
    Symptom = request.form["textfield18"]
    Cause= request.form["textfield19"]
    Remedy= request.form["textfield20"]
    Photo = request.files['filefield']
    Photo.save("C:\\Users\\user\\PycharmProjects\\riceplantdisase\\static\\userimg\\"+Photo.filename)
    url = '/static/userimg/'+Photo.filename
    d = Db()
    s="INSERT INTO adminadd(D_name,Symptom,Cause,Remedy,Photo)VALUES('"+Name+"','"+Symptom+"','"+Cause+"','"+Remedy+"','"+url+"')"
    # print(s)
    res = d.insert(s)
    return '''<script>alert('success');window.location='/admin_add_disease'</script>'''



@app.route('/admin_view_disease')
def admin_view_disease():
    db = Db()
    res = db.select("SELECT * FROM `adminadd`")
    # print(res)
    return render_template('admin/admin_view_symptoms.html',data = res)

@app.route('/admin_delete_disease/<id>')
def admin_delete_disease(id):
    db=Db()
    res=db.delete("DELETE FROM adminadd WHERE D_id ='"+id+"'")


    return '''<script>alert('delete');window.location='/admin_view_disease'</script>'''


@app.route('/rateview')
def rateview():
    db2 = Db()
    res = db2.select("SELECT * FROM `rating`,`user` WHERE `rating`.`L_id`=`user`.`lid`")

    # print(res)
    return render_template('admin/rateview.html',dq = res)




@app.route('/view_registeredusers')
def view_registeredusers():
    db1 = Db ()
    res = db1.select("SELECT * FROM `user`")
    # print(res)
    return render_template('admin/view_registeredusers.html',ds=res)


# ------------------------TEMPLATES------------------------


@app.route('/log')
def login_index():
    return render_template('login_index.html')


@app.route('/admin_index')
def admin_index():
    return render_template('admin/admin_index.html')


@app.route('/user_index')
def user_index():
    return render_template('user/user_index.html')

@app.route('/registration')
def register_index():
    return render_template('registration_index.html')

@app.route('/')
def homeindex():
    return render_template('homeindex.html')


if __name__ == '__main__':
    app.run(debug=True, port=1110)
