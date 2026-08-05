from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='RunResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('country', models.CharField(db_index=True, max_length=2)),
                ('elec_zone', models.CharField(blank=True, max_length=8, null=True)),
                ('h2_zone', models.CharField(blank=True, max_length=8, null=True)),
                ('start_day', models.IntegerField()),
                ('end_day', models.IntegerField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('elec_metrics', models.JSONField(blank=True, null=True)),
                ('h2_metrics', models.JSONField(blank=True, null=True)),
                ('price_series', models.JSONField(blank=True, default=dict)),
                ('exchange', models.JSONField(blank=True, null=True)),
                ('lp_error', models.TextField(blank=True, null=True)),
            ],
            options={
                'ordering': ['-created_at'],
                'indexes': [models.Index(fields=['country', '-created_at'], name='proxy_dashb_country_78c588_idx')],
            },
        ),
    ]
