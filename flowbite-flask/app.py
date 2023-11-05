from flask import Flask, render_template, send_from_directory, url_for
from flask_uploads import IMAGES, UploadSet, configure_uploads
import os
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
from imagedetection import *
from PIL import Image

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)
app.config["UPLOADED_PHOTOS_DEST"] = "uploads"

photos = UploadSet("photos", IMAGES)
configure_uploads(app, photos)

class UploadPhoto(FlaskForm):
	photo = FileField(
		validators=[
			FileAllowed(photos, 'Only images allowed'),
			FileRequired('File field should not be empty')
		]
	)
	submit = SubmitField('Upload')

@app.route('/uploads/<filename>')
def get_file(filename):
	return send_from_directory(app.config["UPLOADED_PHOTOS_DEST"], filename)


@app.route("/", methods=['GET', 'POST'])
def upload_image():
	form = UploadPhoto()
	snake_name = None
	if form.validate_on_submit():
		filename = photos.save(form.photo.data)
		file_url = url_for('get_file', filename=filename)
		print(filename)
		print(file_url)
		print(os.getcwd())
		loader = f"{os.getcwd()}{file_url}"
		print(loader)
		snake_name = image_recognition(loader)
	else:
		file_url = None
	if snake_name is not None:
		return render_template('index.html', form=form, file_url=file_url, snake_name=snake_name)
	else:
		return render_template('index.html', form=form, file_url=file_url)




if __name__ == '__main__':
	app.run(debug=True, port=5001)

