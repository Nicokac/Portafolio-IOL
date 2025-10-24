import os
from pathlib import Path

from infrastructure.iol.client import IOLClient
from application.portfolio_service import PortfolioService, detect_bond_scale_anomalies
from services.data_fetch_service import get_portfolio_data_fetch_service
from services.portfolio_view import PortfolioViewModelService
from domain.models import Controls


def main() -> None:
    print("🔑 Inicializando cliente IOL...")
    cli = IOLClient(
        user=os.environ["IOL_USERNAME"],
        password=os.environ["IOL_PASSWORD"],
        tokens_file=Path(os.environ.get("IOL_TOKENS_FILE", "tokens/iol_tokens.json")),
    )

    print("📦 Cargando servicios de portafolio...")
    psvc = PortfolioService()
    dataset_service = get_portfolio_data_fetch_service()

    print("⬇️ Descargando dataset desde IOL (o cache local)...")
    dataset, meta = dataset_service.get_dataset(cli, psvc)

    print("🧮 Generando vista del portafolio...")
    controls = Controls()
    view_service = PortfolioViewModelService()
    snapshot = view_service.get_portfolio_view(
        df_pos=dataset.positions,
        controls=controls,
        cli=cli,
        psvc=psvc,
    )

    print("🔍 Ejecutando análisis de escala de bonos/letras...")
    report_df, total_impact = detect_bond_scale_anomalies(snapshot.df_view)

    print("\n📊 Resultado del análisis de escala:\n")
    print(report_df.to_string(index=False))
    print(f"\n💰 Impacto total estimado: {total_impact:,.2f} ARS")


if __name__ == "__main__":
    main()
