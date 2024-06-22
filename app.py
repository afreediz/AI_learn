from flask import Flask, render_template, redirect, url_for, flash, request
from config import Config
from db.item import db, Item
from forms import ItemForm

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    items = Item.query.all()
    return render_template('read.html', items=items)

@app.route('/create', methods=['GET', 'POST'])
def create():
    form = ItemForm()
    if form.validate_on_submit():
        item = Item(name=form.name.data, description=form.description.data)
        db.session.add(item)
        db.session.commit()
        flash('Item created successfully!', 'success')
        return redirect(url_for('index'))
    return render_template('create.html', form=form)

@app.route('/update/<int:item_id>', methods=['GET', 'POST'])
def update(item_id):
    item = Item.query.get_or_404(item_id)
    form = ItemForm(obj=item)
    if form.validate_on_submit():
        item.name = form.name.data
        item.description = form.description.data
        db.session.commit()
        flash('Item updated successfully!', 'success')
        return redirect(url_for('index'))
    return render_template('update.html', form=form)

@app.route('/delete/<int:item_id>', methods=['POST'])
def delete(item_id):
    item = Item.query.get_or_404(item_id)
    db.session.delete(item)
    db.session.commit()
    flash('Item deleted successfully!', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
