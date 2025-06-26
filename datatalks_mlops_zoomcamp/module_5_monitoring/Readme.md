## Evidently for generating reports and Grafana for visualization of metrics and Adminer as SQL database admin ui

### Run script to generate dummy metrics into Postgresql and use Grafana for visualization of these metrics

1. Run docker compose to run Grafana and Adminer
```bash
cd docker_compose
docker compose up
```
2. Run python script [dummy_metrics_calculation.py](./docker_compose/dummy_metrics_calculation.py) to generate metrics for Grafana in Postgresql
```bash
cd docker_compose
python dummy_metrics_calculation.py
```
3. Connect to Grafana on [localhost:3000](http://localhost:3000)
4. Create new Dashboard and add new Panel on it
5. Select Postgresql as Data source
 

Grafana dummy_metrics panel as Time Series visualization based on Postgresql as Data source:

```sql
SELECT
  time,
  value1
FROM dummy_metrics
WHERE $__timeFilter(time)
ORDER BY time
```


### Run script to generate evidently metrics into Postgresql and use Grafana for visualization of these metrics

1. Run docker compose to run Grafana and Adminer
```bash
cd docker_compose
docker compose up
```
1. Run python script [evidently_metrics_calculation.py](./evidently/evidently_metrics_calculation.py) to generate metrics for Grafana in Postgresql
```bash
cd evidently
python evidently_metrics_calculation.py
```
1. Connect to Grafana on [localhost:3000](http://localhost:3000)
2. Create new Dashboard and add new Panel on it
3. Select Postgresql as Data source

Grafana dummy_metrics panel as Time Series visualization based on Postgresql as Data source:

```sql
SELECT
  time,
  prediction_drift
FROM evidently_metrics
WHERE $__timeFilter(time)
ORDER BY time
```